#!/usr/bin/env python3
"""Realistic, parameterized acceptance tests for the A40-inspired teaching model.

This intentionally does NOT claim A40 performance validity. It checks that
learners' actions, observable instrumentation, and the model's stated
assumptions behave coherently across scenarios and adversarial parameters.
"""
import os
import random
import re
import subprocess
import sys
import unittest
from pathlib import Path

EXE=Path(os.environ.get('A40_BIN',Path(__file__).with_name('a40_model')))

def run(*args, ok=0):
    proc=subprocess.run([str(EXE),*map(str,args)],capture_output=True,text=True,timeout=8)
    if proc.returncode != ok:
        raise AssertionError(f'Unexpected rc={proc.returncode} expected {ok}: {args}\n{proc.stdout[-1800:]}\n{proc.stderr[-1800:]}')
    return proc.stdout

def metric(output, label):
    m=re.search(r'^\s*'+re.escape(label)+r'\s+([0-9.]+)',output,re.M)
    if not m:raise AssertionError(f'Missing metric {label}')
    return float(m.group(1))

class ModelTests(unittest.TestCase):
    def test_realistic_memory_scenarios(self):
        # Same memory workload. Change ONLY lane spacing; verify 32-byte
        # sector requests increase from aligned words to stride-2, to scatter.
        per_access={0:4,1:8,2:32,4:1,5:18}
        scenarios={}
        for pattern,expected in per_access.items():
            text=run('--sms',1,'--warps',1,'--instructions',48,
                     '--workload','memory','--pattern',pattern,'--quiet')
            self.assertEqual(metric(text,'Total Instructions Issued:'),48)
            self.assertEqual(metric(text,'Avg transactions/access:'),expected)
            self.assertEqual(metric(text,'Total memory transactions:'),48*expected)
            scenarios[pattern]=metric(text,'Total Cycles:')
        self.assertGreater(scenarios[2],scenarios[0])
        # Timing relation is only a toy-model prediction, not measured A40.

    def test_compute_only_ignores_address_pattern_and_scales_work(self):
        times=[]
        for pattern in (0,2,4):
            text=run('--sms',2,'--warps',3,'--instructions',32,
                     '--workload','compute','--pattern',pattern,'--quiet')
            self.assertEqual(metric(text,'Total Instructions Issued:'),192)
            self.assertEqual(metric(text,'Total memory transactions:'),0)
            times.append(metric(text,'Total Cycles:'))
        self.assertEqual(len(set(times)),1)

    def test_mixed_workload_exposes_branch_and_memory_in_one_run(self):
        text=run('--sms',2,'--warps',3,'--instructions',128,
                 '--workload','mixed','--pattern',5,'--quiet')
        self.assertGreater(metric(text,'Total branches:'),0)
        self.assertGreater(metric(text,'Divergent branches:'),0)
        self.assertGreater(metric(text,'Total memory transactions:'),0)
        self.assertIn('Termination: complete',text)

    def test_cycle_budget_and_bad_input_dont_fake_success(self):
        text=run('--sms',1,'--warps',3,'--instructions',128,
                 '--workload','memory','--cycles',10,'--quiet',ok=4)
        self.assertIn('Termination: cycle limit',text)
        p=subprocess.run([str(EXE),'--warps','900'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)

    def test_parameterized_memory_probes_against_independent_address_arithmetic(self):
        r=random.Random(0xA40)
        cases=[(0,4,0xffffffff),(96,4,0xffffffff),(4,4,0xffffffff),
               (0,128,0xffffffff),(0,0,0xffffffff),(0,4,0),
               (31,0,0xffffffff),(31,64,0xffffffff),(30,4,0xffffffff)]
        cases.extend((r.randrange(0,2048,4),r.randrange(0,200,4),r.getrandbits(32)) for _ in range(140))
        for base,stride,mask in cases:
            out=run('--probe','--base',base,'--stride',stride,'--mask',hex(mask))
            rows=[]
            for line in out.splitlines():
                if re.fullmatch(r'\d+,\d+,\d+,\d+',line):
                    t,addr,sector,cache_line=map(int,line.split(','));rows.append((t,addr,sector,cache_line))
            expected=[(t,base+t*stride,(base+t*stride)//32,(base+t*stride)//128)
                      for t in range(32) if mask & (1<<t)]
            self.assertEqual(rows,expected,(base,stride,mask))
            distinct={sector for _,a,_,_ in expected for sector in range(a//32,(a+3)//32+1)}
            self.assertIn(f'sectors={len(distinct)};',out)

    def test_branch_predicates_with_partial_masks(self):
        r=random.Random(109)
        funcs=[lambda t:t<16,lambda t:t>=16,lambda t:t%2==0,
               lambda t:t%4<2,lambda t:t<8 or t>=24,lambda t:True,
               lambda t:False,lambda t:(t*7+13)%3==0]
        for pred in range(8):
            for mask in [0,0xffffffff,0x55555555,0x80000000,r.getrandbits(32)]:
                out=run('--branch-probe','--predicate',pred,'--mask',hex(mask))
                rows=[tuple(map(int,line.split(','))) for line in out.splitlines()
                      if re.fullmatch(r'\d+,[01]',line)]
                target=[(t,int(funcs[pred](t))) for t in range(32) if mask & (1<<t)]
                self.assertEqual(rows,target)
                count=sum(x[1] for x in target)
                self.assertIn(f'active={len(target)}; taken={count};',out)
                self.assertIn('divergent='+('yes' if 0<count<len(target) else 'no'),out)

    def test_cache_lru_after_real_conflict(self):
        # Five aligned addresses with the same L1 set. The fifth access must
        # evict the actual LRU way; re-reading the most-recent line must hit.
        out=run('--cache-seq','0,32768,65536,98304,0,131072,32768,0')
        outcomes=[line.split(',')[2] for line in out.splitlines()
                  if re.fullmatch(r'\d+,\d+,(?:hit|miss),\d+,\d+',line)]
        self.assertEqual(outcomes,['miss','miss','miss','miss','hit','miss','miss','hit'])

if __name__=='__main__':
    unittest.main(verbosity=2)

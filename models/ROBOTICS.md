# Robot mathematics: joint uncertainty and scalar observations

The standalone [robotics.html](../robotics.html) contains a reusable two-link planar kinematics and uncertainty model.

With joint configuration q, forward kinematics p=f(q) and Jacobian J(q), it calculates the first-order approximation Σp≈JΣqJᵀ. Joint covariance includes adjustable standard deviations and correlation; after a noisy observation of x or y, a linear-Gaussian covariance update P′=P−Phᵀ(hPhᵀ+R)⁻¹hP reduces the predicted uncertainty.

The SVG overlays 2σ ellipses and exposes J, Σq, Σp and P′. The virtual engineering mission adds a specified target pose, a 9 cm nominal position tolerance, a 6 cm largest 1σ covariance bound and a maximum of two x/y observations. Learners change joint angles, plan observations and verify both constraints; a dedicated mission view displays the target and updated uncertainty. Tests cover successful and unsuccessful observation sequences, out-of-target poses and the order-independence of independent linear-Gaussian observation updates. The older short tasks continue to test scaling, singularity, measurement coupling and correlated errors.

**Scope:** The uncertainty projection is a local linearization; for large angular uncertainty or at a singularity, the true position distribution can depart from a Gaussian ellipse. The sensor covariance update assumes independent Gaussian sensor noise and a direct scalar Cartesian observation. This is not a contact or whole-robot physics engine.

**Reuse:** [Arm Playground](https://bakulbadwal.github.io/arm-playground/) covers kinematics and control exercises; [Robot Localization, Explained](https://ferrolho.github.io/robot-localization-explained/) covers estimation and sensor fusion. This original lab is specifically their intersection, not a replacement.

**Validation:** Cross-check Jacobian columns against independently perturbed forward kinematics (central finite differences); verify determinant identity det J=l1*l2*sin(q2), PSD and monotonicity of covariance updates and the covariance-area scaling law.
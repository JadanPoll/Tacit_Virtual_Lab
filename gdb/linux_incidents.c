/* Reproducible Linux debugging incidents. Build with -g3 -O0 -pthread. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static void usage(const char *name) {
    fprintf(stderr, "Usage: %s {frame|watch|abort|threads|syscall}\n", name);
}

/* Report processing. */
static NOINLINE int sum_records(const int *records, size_t record_count) {
    int sum = 0;
    for (size_t i = 0; i <= record_count; ++i) {
        sum += records[i];
    }
    return sum;
}
static NOINLINE int prepare_report(void) {
    const int records[] = {3, 5, 7, 11, 97};
    const size_t valid_records = 4;
    return sum_records(records, valid_records);
}
static void frame_incident(void) {
    const int got = prepare_report();
    printf("Report total: %d (expected 26)\n", got);
}

/* Session state and packet processing. */
struct session { int max_requests; int requests_seen; };
static struct session session_state;
static NOINLINE void accept_packet(unsigned opcode) {
    session_state.requests_seen++;
    if (opcode == 0x42U) {
        session_state.max_requests = session_state.requests_seen;
    }
}
static NOINLINE void init_session(void) {
    session_state.max_requests = 32;
    session_state.requests_seen = 0;
}
static void watch_incident(void) {
    init_session();
    const unsigned packets[] = {0x10U, 0x11U, 0x42U, 0x13U};
    for (size_t i = 0; i < sizeof packets / sizeof packets[0]; ++i) {
        accept_packet(packets[i]);
    }
    printf("Session max_requests: %d (expected 32)\n", session_state.max_requests);
}

/* Message validation. */
struct message { size_t declared_length; size_t bytes_received; };
static NOINLINE void enforce_message_contract(const struct message *m) {
    if (m->declared_length > m->bytes_received) {
        fprintf(stderr, "fatal: declared payload length %zu exceeds received %zu\n",
                m->declared_length, m->bytes_received);
        abort();
    }
}
static NOINLINE void decode_message(const struct message *m) {
    enforce_message_contract(m);
}
static NOINLINE void receive_message(void) {
    const struct message m = {.declared_length = 7, .bytes_received = 5};
    decode_message(&m);
}
static void abort_incident(void) { receive_message(); }

/* Worker threads. */
static pthread_mutex_t lock_a = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lock_b = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t both_holding_first_lock;
static void *worker_ab(void *arg) {
    (void)arg;
    pthread_mutex_lock(&lock_a);
    pthread_barrier_wait(&both_holding_first_lock);
    pthread_mutex_lock(&lock_b);
    pthread_mutex_unlock(&lock_b);
    pthread_mutex_unlock(&lock_a);
    return NULL;
}
static void *worker_ba(void *arg) {
    (void)arg;
    pthread_mutex_lock(&lock_b);
    pthread_barrier_wait(&both_holding_first_lock);
    pthread_mutex_lock(&lock_a);
    pthread_mutex_unlock(&lock_a);
    pthread_mutex_unlock(&lock_b);
    return NULL;
}
static void threads_incident(void) {
    pthread_t a, b;
    if (pthread_barrier_init(&both_holding_first_lock, NULL, 2) != 0) {
        fprintf(stderr, "barrier initialization failed\n");
        exit(2);
    }
    if (pthread_create(&a, NULL, worker_ab, NULL) != 0 ||
        pthread_create(&b, NULL, worker_ba, NULL) != 0) {
        fprintf(stderr, "could not create worker threads\n");
        exit(2);
    }
    puts("Two workers started; waiting for completion...");
    fflush(stdout);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    pthread_barrier_destroy(&both_holding_first_lock);
}

/* Service startup. */
static NOINLINE int load_config(void) {
    const char *filename = "./tacit_lab_missing_configuration.ini";
    const int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "configuration unavailable: %s (errno=%d)\n", strerror(errno), errno);
        return -1;
    }
    close(fd);
    return 0;
}
static NOINLINE int initialize_service(void) { return load_config(); }
static void syscall_incident(void) {
    const int ok = initialize_service();
    printf("Service startup: %s\n", ok == 0 ? "ready" : "failed");
}

int main(int argc, char **argv) {
    if (argc != 2) { usage(argv[0]); return 2; }
    if (strcmp(argv[1], "frame") == 0) frame_incident();
    else if (strcmp(argv[1], "watch") == 0) watch_incident();
    else if (strcmp(argv[1], "abort") == 0) abort_incident();
    else if (strcmp(argv[1], "threads") == 0) threads_incident();
    else if (strcmp(argv[1], "syscall") == 0) syscall_incident();
    else { usage(argv[0]); return 2; }
    return 0;
}

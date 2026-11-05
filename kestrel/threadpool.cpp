#include "threadpool.hpp"
#include <cstdlib>

static int n_threads() {
  static const int n = [] {
    if (const char *env = std::getenv("KESTREL_THREADS")) {
      return std::max(1, std::atoi(env));
    }
    return (int)std::max(1u, std::thread::hardware_concurrency());
  }();
  return n;
}

ThreadPool::ThreadPool(int n) {
  for (int i = 1; i < n; i++) {
    workers.emplace_back([this, i] { worker_loop(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(m);
    stop = true;
    generation++;
  }
  wake.notify_all();
  for (auto &w : workers) {
    w.join();
  }
}

void ThreadPool::run(const std::function<void(int)> &job) {
  {
    std::lock_guard<std::mutex> lock(m);
    current = &job;
    remaining = (int)workers.size();
    generation++;
  }
  wake.notify_all();
  job(0);
  std::unique_lock<std::mutex> lock(m);
  done.wait(lock, [&] { return remaining == 0; });
}

void ThreadPool::worker_loop(int id) {
  int seen = 0;
  while (true) {
    const std::function<void(int)> *job;
    {
      std::unique_lock<std::mutex> lock(m);
      wake.wait(lock, [&] { return generation != seen; });
      seen = generation;
      if (stop)
        return;
      job = current;
    }
    (*job)(id);
    {
      std::lock_guard<std::mutex> lock(m);
      if (--remaining == 0) {
        done.notify_one();
      }
    }
  }
}

ThreadPool &pool() {
  static ThreadPool p(n_threads());
  return p;
}

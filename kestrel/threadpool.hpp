#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:
  explicit ThreadPool(int n);
  ~ThreadPool();

  int size() const { return (int)workers.size() + 1; };

  void run(const std::function<void(int)> &job);

private:
  void worker_loop(int id);

  std::vector<std::thread> workers;
  std::condition_variable wake, done;
  std::mutex m;
  int remaining = 0;
  int generation = 0;
  bool stop = false;
  const std::function<void(int)> *current = nullptr;
};

ThreadPool &pool();

template <typename F> void parallel_for(int n, F work) {
  ThreadPool &p = pool();
  const int chunk = (n + p.size() - 1) / p.size();
  p.run([&](int i) {
    const int begin = i * chunk;
    const int end = std::min(n, begin + chunk);
    if (begin < end) {
      work(begin, end);
    }
  });
}

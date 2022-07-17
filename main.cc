#include <chrono>
#include <future>
#include <thread>
#include <glog/logging.h>

#include "priority_stream.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "Creating three simulators: \n"
            << "  Sim low_1k_1 is 1000*1000 matmul, low priority 0;\n"
            << "  Sim low_1k_2 is 1000*1000 matmul, low priority 0;\n"
            << "  Sim high_1k is 1000*1000 matmul, high priority -5;\n"
            << "  Sim high_2k is 1000*1000 matmul, high priority -5;\n";

  PriorityStreamSimulator low_1k_1(0, 1000, "low_1k_1");
  PriorityStreamSimulator low_1k_2(0, 1000, "low_1k_2");
  PriorityStreamSimulator high_1k(-5, 1000, "high_1k");
  PriorityStreamSimulator high_2k(-5, 1000, "high_2k");

  auto run_all = [&](const int num_iters, const int sleep_ms, const int sleep_between_runs) {
    LOG(INFO) << "Preheating GPU.";
    low_1k_1.Run(num_iters, false, sleep_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));
    LOG(INFO) << "Running low_1k_1 alone, non-async";
    low_1k_1.RunAndLog(num_iters, false, sleep_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));

    LOG(INFO) << "Running low_1k_1 alone, async";
    low_1k_1.RunAndLog(num_iters, true, sleep_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));

    LOG(INFO) << "Running high_1k alone, non-async";
    high_1k.RunAndLog(num_iters, false, sleep_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));

    LOG(INFO) << "Running high_1k alone, async";
    high_1k.RunAndLog(num_iters, true, sleep_ms);
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));

    LOG(INFO) << "Running low_1k_1 and low_1k_2, non-async, at the same time";
    {
      auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_1, num_iters, false, sleep_ms);
      auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_2, num_iters, false, sleep_ms);
      a1.wait();
      a2.wait();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));

    LOG(INFO) << "Running low_1k_1 and low_1k_2, async, at the same time";
    {
      auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_1, num_iters, true, sleep_ms);
      auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_2, num_iters, true, sleep_ms);
      a1.wait();
      a2.wait();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));


    LOG(INFO) << "Running low_1k_1 and high_1k, non-async, at the same time";
    {
      auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_1, num_iters, false, sleep_ms);
      auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &high_1k, num_iters, false, sleep_ms);
      a1.wait();
      a2.wait();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));

    LOG(INFO) << "Running low_1k_1 and high_1k, async, at the same time";
    {
      auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_1, num_iters, true, sleep_ms);
      auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &high_1k, num_iters, true, sleep_ms);
      a1.wait();
      a2.wait();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_between_runs));
  };

  //LOG(INFO) << "==== test continuous run ====";
  //run_all(10, 0);
  //LOG(INFO) << "==== test run where QPS=1000 ====";
  //run_all(100, 1);

  // Per Xiang: 
  // Running low-pri, always.
  // Running high-pri for a couple seconds.
  // Continue Running low-pri.
  //
  // My modified schedule:
  // Running low-pri for 500 iters, which should be long enough.
  // main thread sleep for 10 milliseconds for low-pri to have a headstart
  // Running high-pri for 50 iters
  // Wait for both to finish.
  {
    auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_1k_1, 500, true, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &high_1k, 50, true, 0);
    a1.wait();
    a2.wait(); 
  }

	return 0;
}
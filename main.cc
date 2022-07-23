#include <chrono>
#include <future>
#include <thread>
#include <glog/logging.h>

#include "priority_stream.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  {
    LOG(INFO) << "==== GO 3 ====";
    PriorityStreamSimulator low_10k_1(0, 10000, "low_10k_1");
    PriorityStreamSimulator high_10k_1(-2, 10000, "high_10k_1");
    PriorityStreamSimulator high_10k_2(-2, 10000, "high_10k_2");
    auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_10k_1, 100, true, 0);
    auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_1, 100, true, 0);
    auto a3 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_2, 100, true, 0);


    a1.wait();
    a2.wait(); 
    a3.wait();
  }


  {
    LOG(INFO) << "==== GO 6 ====";
    PriorityStreamSimulator low_10k_1(0, 10000, "low_10k_1");
    PriorityStreamSimulator high_10k_1(-2, 10000, "high_10k_1");
    PriorityStreamSimulator high_10k_2(-2, 10000, "high_10k_2");
    PriorityStreamSimulator high_10k_3(-2, 10000, "high_10k_3");
    PriorityStreamSimulator high_10k_4(-2, 10000, "high_10k_4");
    PriorityStreamSimulator high_10k_5(-2, 10000, "high_10k_5");
    auto a1 = std::async(&PriorityStreamSimulator::RunAndLog, &low_10k_1, 100, true, 0);
    auto a2 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_1, 100, true, 0);
    auto a3 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_2, 100, true, 0);
    auto a4 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_3, 100, true, 0);
    auto a5 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_4, 100, true, 0);
    auto a6 = std::async(&PriorityStreamSimulator::RunAndLog, &high_10k_5, 100, true, 0);


    a1.wait();
    a2.wait(); 
    a3.wait();
    a4.wait();
    a5.wait(); 
    a6.wait(); 
  }

	return 0;
}

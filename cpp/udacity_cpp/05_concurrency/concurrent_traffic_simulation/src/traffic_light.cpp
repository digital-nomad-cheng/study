#include <iostream>
#include <random>
#include <future>
#include "traffic_light.hpp"

/* Implementation of class "MessageQueue" */

template <typename T>
T MessageQueue<T>::receive()
{
    // FP.5a : The method receive should use std::unique_lock<std::mutex> and 
    // _condition.wait() to wait for and receive new messages and pull them from
    // the queue using move semantics. The received object should then be 
    // returned by the receive function.
    std::unique_lock<std::mutex> ulock(_mutex);
    _condition.wait(ulock, [this] { return !_queue.empty(); });
    T msg = std::move(_queue.front());
    _queue.pop_front();
    return msg;
}

template <typename T>
void MessageQueue<T>::send(T &&msg)
{
    // FP.4a : The method send should use the mechanisms 
    // std::lock_guard<std::mutex> as well as _condition.notify_one() to add a 
    // new message to the queue and afterwards send a notification.
    std::lock_guard<std::mutex> guard(_mutex);
    _queue.push_back(std::move(msg));
    _condition.notify_one();
}

/* Implementation of class "TrafficLight" */

TrafficLight::TrafficLight()
{
    _current_phase = TrafficLightPhase::red;
    _message_queue = std::make_shared<MessageQueue<TrafficLightPhase>>();
}

void TrafficLight::waitForGreen()
{
    // FP.5b : add the implementation of the method waitForGreen, in which an 
    // infinite while-loop  runs and repeatedly calls the receive function on 
    // the message queue. Once it receives TrafficLightPhase::green, the method 
    // returns.
    while(1) {
        std::this_thread::sleep_for( std::chrono::milliseconds(1) );
        TrafficLightPhase phase = _message_queue->receive();
        if (phase == TrafficLightPhase::green) {
            return;
        }
    }
}

TrafficLightPhase TrafficLight::getCurrentPhase()
{
    return _current_phase;
}

void TrafficLight::simulate()
{
    // FP.2b : Finally, the private method "cycleThroughPhases" should be started
    // in a thread when the public method "simulate" is called. To do this, use 
    // the thread queue in the base class. 
    threads.emplace_back(std::thread(&TrafficLight::cycleThroughPhases, this));
}

// virtual function which is executed in a thread
void TrafficLight::cycleThroughPhases()
{
    // FP.2a : Implement the function with an infinite loop that measures the 
    // time between two loop cycles and toggles the current phase of the traffic
    // light between red and green and sends an update method to the message 
    // queue using move semantics. The cycle duration should be a random value 
    // between 4 and 6 seconds. Also, the while-loop should use 
    // std::this_thread::sleep_for to wait 1ms between two cycles. 
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distribution(4, 6);
    std::unique_lock<std::mutex> lock(_mutex);
    std::cout << "Traffic Light #" << _id << "::cycleThroughtPhases: thread id = " << std::this_thread::get_id() << std::endl;
    lock.unlock();

    int cycle_duration = distribution(eng);
    auto start = std::chrono::system_clock::now();
    while(1) {
        auto end = std::chrono::system_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        long elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        if (elapsed_seconds > cycle_duration) {
            std::lock_guard guard(_mutex);
            std::cout << "elapsed_seconds: " << elapsed_seconds 
                      << "cycle_duration: " << cycle_duration << std::endl;
            switch (_current_phase) {
                case TrafficLightPhase::red:
                    _current_phase = TrafficLightPhase::green;
                    // std::cout << "red to green" << std::endl;
                    break;
                case TrafficLightPhase::green:
                    _current_phase = TrafficLightPhase::red;
                    // std::cout << "green to red" << std::endl;
                    break;
            }
            start = std::chrono::system_clock::now();
            cycle_duration = distribution(eng);
            // _message_queue->send(std::move(_current_phase));
            auto msg = _current_phase;
            auto is_sent = std::async( std::launch::async, &MessageQueue<TrafficLightPhase>::send, _message_queue, std::move(msg) );
            is_sent.wait();
        }
        
    }
}


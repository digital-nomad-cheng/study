#include <iostream>
#include <thread>
#include <future>

void modifyMessage(std::promise<std::string> &&prms, std::string message)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(4000));
    std::string modified_message = message + " has been modified";
    prms.set_value(modified_message);
}

int main()
{
    std::string message_to_thread = "My Message";
    std::promise<std::string> prms;
    std::future<std::string> ftr = prms.get_future();
    
    std::thread t(modifyMessage, std::move(prms), message_to_thread);

    std::cout << "Original message from main(): " << message_to_thread 
              << std::endl;
    
    // retrieve modified message via future and print to console
    std::string message_from_thread = ftr.get();
    std::cout << "Modified message from thread(): " << message_from_thread << std::endl;

    // thread barrier
    t.join();

    return 0;
}


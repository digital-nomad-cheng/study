#include <iostream>
#include <string>

class BankAccount
{
public: 
    BankAccount(int num, std::string name, float fund)
    {
        this->account_number = num;
        this->owner_name = name;
        this->fund = fund;
    }
    
    int getAccountNum() const
    {
        return this->account_number;
    }

    std::string getOwnerName() const
    {
        return this->owner_name;
    }

    float getFund() const
    {
        return this->fund;
    }

    void setAccountNum(int num)
    {
        this->account_number = num;
    }
    
    void setOwnerName(std::string name)
    {
        this->owner_name = name;
    }

    void setFund(float fund)
    {
        this->fund = fund;
    }

    
private:
    int account_number;
    std::string owner_name;
    float fund;
};

int main()
{
    BankAccount ba(0, "vincent", 1000);
    std::cout << "Account fund: " << ba.getFund() << std::endl;
}


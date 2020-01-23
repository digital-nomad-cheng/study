#include <iostream>
#include <string>

using namespace std;

class Car
{
public:
   int getHorsePower() const
   {
       return this->horse_power;
   }
   int getWeight() const
   {
       return this->weight;
   }
   string getBrand() const
   {
       std::string brand_name = "Brand name:";
       brand_name += this->brand;
       return brand_name;
   }
   void setHorsePoer(int horse_power)
   {
       this->horse_power = horse_power;
   }
   void setWeight(int weight)
   {
       this->weight = weight;
   }
   void setBrand(string brand_name)
   {
       this->brand = new char[brand_name.length()+1];
       strcpy(this->brand, brand_name.c_str());
   } 

private:
    int horse_power;
    int weight;
    char *brand;    
};

int main()
{
    Car car;
    car.setBrand("peugeot");
    std::cout << car.getBrand() << "\n";
    return 0;
}


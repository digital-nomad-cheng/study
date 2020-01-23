#include <iostream>
#include <string>

class Pyramid
{
public:
    Pyramid(int length, int width, int height)
    {
        this->base_length = length;
        this->base_width = width;
        this->height = height;
    }

    int getLength() const
    {
        return this->base_length;
    }
    int getWidth() const
    {
        return this->base_width;
    }
    int getHeight() const
    {
        return this->height;
    }

    void setLength(int length)
    {
        this->base_length = length;
    }
    void setWidth(int width)
    {
        this->base_width = width;
    }
    void setHeight(int height)
    {
        this->height = height;
    }
    
    int getVolume() const
    {
        return calculateVolume();
    }
    
    int getArea() const
    { 
        return calculateArea();
    }
private:
   
   int calculateVolume() const
   {
        checkValid();
        int area = base_length * base_width;
        int volume = area*height/3;
        return volume;
   }

   int calculateArea() const
   {
       throw std::runtime_error("Not Implemented"); 
   }
   void checkValid() const
   {
       if (this->base_length <= 0 || this->base_width <= 0 || this->height <=0) {
           throw std::domain_error("Value must be positive");
       }
       return;
   }

   int base_length;
   int base_width;
   int height;
};

int main()
{
    Pyramid pyr(20, 20, 20);
    std::cout << "volume:" << pyr.getVolume() << std::endl;
    pyr.getArea();
}


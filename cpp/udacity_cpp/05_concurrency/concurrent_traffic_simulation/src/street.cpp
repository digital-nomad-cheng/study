#include <iostream>
#include "vehicle.hpp"
#include "intersection.hpp"
#include "street.hpp"

Street::Street()
{
    _type = ObjectType::objectStreet;
    _length = 1000.0; // in m
}

void Street::setInIntersection(std::shared_ptr<Intersection> in)
{
    _interIn = in;
    // add this street to list of streets connected to the intersection 
    in->addStreet(get_shared_this()); 
}

void Street::setOutIntersection(std::shared_ptr<Intersection> out)
{
    _interOut = out;
    // add this street to list of streets connected to the intersection
    out->addStreet(get_shared_this()); 
}

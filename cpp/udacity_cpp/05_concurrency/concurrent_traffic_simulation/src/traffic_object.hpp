#ifndef TRAFFICOBJECT_H
#define TRAFFICOBJECT_H

#include <vector>
#include <thread>
#include <mutex>

enum ObjectType
{
    noObject,
    objectVehicle,
    objectIntersection,
    objectStreet,
};

class TrafficObject
{
public:
    // constructor / desctructor
    TrafficObject();
    ~TrafficObject();

    // getter and setter
    int getID() { return _id; }
    void setPosition(double x, double y);
    void getPosition(double &x, double &y);
    ObjectType getType() { return _type; }

    // typical behaviour methods
    virtual void simulate(){};

protected:
    ObjectType _type; 
    int _id; // every traffic object has its own unique id
    double _posX, _posY;              
    std::vector<std::thread> threads; 
    static std::mutex _mtx; // mutex shared by all traffic objects for protecting cout 

private:
    static int _idCnt; // global variable for counting object ids
};

#endif

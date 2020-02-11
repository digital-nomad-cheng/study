
CC=gcc
CXX=g++
LN=g++
VERSION=1.0
NAME=libmxgc

CXXFLAGS=-std=c++11 -O3 -pthread

STATICLIB=$(NAME).a
SHAREDLIB=$(NAME).so.$(VERSION)
SONAME:=$(shell echo $(SHAREDLIB) | sed -re 's/\.[0-9]+$$//')
SOLIB:=$(shell echo $(SHAREDLIB) | sed -re 's/\.[0-9]+//g')

PREFIX?=/usr
LIBPREFIX?=$(PREFIX)/lib

.PHONY: all sharedlib staticlib install unittest dist doc

all: gc_test sharedlib staticlib

sharedlib: $(SHAREDLIB)

staticlib: $(STATICLIB)

gc_test:   gc_test.cxx gc_ptr.cxx gc_ptr.h
	$(CXX) $(CXXFLAGS) -o $@ gc_test.cxx gc_ptr.cxx

$(SHAREDLIB): gc_ptr.cxx gc_ptr.h
	$(CXX) $(CXXFLAGS) -fpic -shared -Wl,-soname=$(SONAME) -o $(SHAREDLIB) gc_ptr.cxx
	-rm -f $(SONAME) $(SOLIB)
	ln -s $(SHAREDLIB) $(SONAME)
	ln -s $(SHAREDLIB) $(SOLIB)

$(STATICLIB): gc_ptr.cxx gc_ptr.h
	$(CXX) $(CXXFLAGS) -c gc_ptr.cxx
	-rm -f $(STATICLIB)
	ar qc $(STATICLIB) gc_ptr.o

install: sharedlib staticlib
	mkdir -p $(PREFIX)/include $(LIBPREFIX)
	cp gc_ptr.h $(PREFIX)/include
	cp $(STATICLIB) $(LIBPREFIX)
	cp $(SHAREDLIB) $(LIBPREFIX)/$(SHAREDLIB)
	-rm -f $(LIBPREFIX)/$(SONAME) $(LIBPREFIX)/$(SOLIB)
	ln -s $(SHAREDLIB) $(LIBPREFIX)/$(SONAME)
	ln -s $(SHAREDLIB) $(LIBPREFIX)/$(SOLIB)

unittest: gc_test
	./gc_test && echo "*** Test passed ***"

doc/html/index.html: gc_ptr.h gc_ptr.dox
	doxygen Doxyfile

doc: doc/html/index.html

dist: doc
	tar caf meixnergc-$(VERSION).tar.gz gc_test.cxx gc_ptr.cxx gc_ptr.h gc_ptr.dox Makefile README Doxyfile Changelog doc

CXX=g++
IDIR=./eigen3
CXXFLAGS=-Wall -g -I$(IDIR) 
TARGET=lasso

# $(wildcard *.cc /xxx/xxx/*.cc): get all .cpp files from the current directory and dir "/xxx/xxx/"
SRCS= $(wildcard *.cc)
# $(patsubst %.cc,%.o,$(SRCS)): substitute all ".cpp" file name strings to ".o" file name strings
OBJS := $(patsubst %.cc,%.o,$(SRCS))

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<
clean:
	rm -rf $(TARGET) *.o
	
.PHONY: all clean


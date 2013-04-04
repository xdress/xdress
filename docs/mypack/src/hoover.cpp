#include "hoover.h"

using namespace hoover;

A::A(int x){y[x] = x * 42.0;};
A::~A(){};

B::B(){z=3;};
B::~B(){};

void do_nothing_ab(A a, B b) {};

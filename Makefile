cxx = g++
prom = batch_pictures
source = batch_pictures.cpp
CXXFLAGS = ${shell pkg-config opencv --cxxflags}
CXXFLAGS +=-O0 -Wall -g2 -ggdb -std=gnu++14
LDFLAGS = ${shell pkg-config opencv --libs} -L/usr/lib/x86_64-linux-gnu/ -lboost_filesystem -lboost_system -lboost_regex -lboost_program_options
$(prom): $(source)
	$(cxx) -o $(prom) $(source) ${CXXFLAGS} ${LDFLAGS}

.python : clean
clean:
	rm $(prom)



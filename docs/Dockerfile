FROM python:3.8

RUN apt install make git
RUN python3 --version

RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz && \
    tar -zxvf cmake-3.15.2.tar.gz && \
    cd cmake-3.15.2 && \
    ./bootstrap && \
    make && \
    make install

RUN pip3 install git+https://github.com/sphinx-doc/sphinx && \
    pip3 install git+https://github.com/sqoshi/mask-imposer.git && \
    pip3 install sphinx-autobuild m2r2 sphinx_rtd_theme
RUN ls
CMD sphinx-autobuild -b html --host 0.0.0.0 --port 80 /mask-imposer/docs/source /mask-imposer/docs/build
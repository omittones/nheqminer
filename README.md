# Build instructions:

## Windows:

Windows builds made by us are available here: https://github.com/nicehash/nheqminer/releases

Download and install:
- Visual Studio 2013 Community: https://www.visualstudio.com/en-us/news/releasenotes/vs2013-community-vs
- Visual C++ Compiler November 2013 CTP: https://www.microsoft.com/en-us/download/details.aspx?id=41151

Open **nheqminer.sln** under **nheqminer/nheqminer.sln** and build.

## Linux cmake **recommended** (Tested on Ubuntu Desktop 14.04 and 16.04 and Ubuntu server 14.04):
You should have **CMake** installed (2.8 minimal version), boost (install from the repositories or download boost manually build and install it manually), download the sources manually or via git. 
Under Ubuntu open a terminal and run the following commands:
  - `sudo apt-get install cmake build-essential libboost-all-dev`
  - `git clone https://github.com/sarath-hotspot/nheqminer.git`
  - `cd nheqminer/nheqminer`
  - `mkdir build`
  - `cd build`
  - `cmake ..`
  - `make`


## Linux (Ubuntu/Debian based, Tested on Ubuntu 16.04):
To build under Ubuntu Linux make sure you have Qt5 installed. You can install it manually from [Qt website](https://www.qt.io/) or install it from the command line: `sudo apt-get install qt5-default`.
Open a terminal and cd to nheqminer root folder and run the following commands (make sure you have qmake in your PATH, if installed manually from Qt website you will have to export it to your PATH):
  - `git clone https://github.com/sarath-hotspot/nheqminer.git`
  - `cd nheqminer`
  - `mkdir build`
  - `cd build`
  - `qmake ../nheqminer/nheqminer.pro`
  - `make`
  

# Run instructions:

If run without parameters, miner will start mining with 75% of available virtual cores on NiceHash. Use parameter -h to learn about available parameters:

        -h              Print this help and quit
        -l [location]   Location (eu, usa, hk, jp)
        -u [username]   Username (bitcoinaddress)
        -p [password]   Password (default: x)
        -t [num_thrds]  Number of threads (default: number of sys cores)
        -d [level]      Debug print level (0 = print all, 5 = fatal only, default: 2)
        -b [hashes]     Run in benchmark mode (default: 100 hashes)
        -a [port]       Local API port (default: 0 = do not bind)
        
Example to run benchmark:

        nheqminer_x64_AVX.exe -b
        
Example to run with full logging (including network dump):

        nheqminer_x64_AVX.exe -d 0
        
Example to mine with your own BTC address and worker1 on USA server:

        nheqminer_x64_AVX.exe -l usa -u YOUR_BTC_ADDRESS_HERE.worker1

Example to mine with your own BTC address and worker1 on EU server, using 6 threads:

        nheqminer_x64_AVX.exe -l eu -u YOUR_BTC_ADDRESS_HERE.worker1 -t 6

<i>Note: if you have a 4-core CPU with hyper threading enabled (total 8 threads) it is best to run with only 6 threads (experimental benchmarks shows that best results are achieved with 75% threads utilized)</i>


# Sarath's Edits

My integration of Xenoncat's code in nheqminer works only in Linux (I did not verify it in other operating systems). 

## Pointing this miner to other stratum based pools. 
### supernova
nheqminer -l zec.suprnova.cc:2142 -u username.workername -p password -t threadCount 

### zmine
nheqminer  -l zmine.io:1337 -u ZcashTransparentAddress


## Donations
If you feel this project is useful to you. Feel free to donate.

    BTC address: 1PqPwYC8u2XCQxJv6qu3HwCRdQkdsqhXRS


# lhl's edits

I switched AVX2 to AVX1 support, see commits. Builds w/ cmake the same way

Running `-b` benchmarks the same, however, in practice I am seeing:

* 5.6 sol/s -> 9.4 sol/s  w/ -t3 on an Intel(R) Core(TM) i3-2130 CPU @ 3.40GHz (4C) ; Ubuntu 16.04
* 12.2 sol/s -> 20.2 sol/s w -t6 on an Intel(R) Core(TM) i7-3770S CPU @ 3.10GHz (4C/8HT) ; Arch

# Image Show Project

This project compares the speed of video rendering between Python and C++.


## Setup

### python

```
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

### C++

```
$ sudo apt install cmake libopencv-dev libfmt-dev build-essential 
```

```
$ mkdir cpp/build
$ cd cpp/build
$ cmake ..
$ make
```

## Rendering time

| Language | Library | Time(msec) |
| :------: | :-----: | :--------: |
|  Python  | OpenCV  |   2.7558   |
|  Python  | OpenGL  |   1.3904   |
|   C++    | OpenCV  |   2.7722   |
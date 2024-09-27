from subprocess import run
from os.path import isfile

if not(isfile("bobyqa.out")):
    run("g++ bobyqa.cpp -o bobyqa.out -lnlopt".split(" "))

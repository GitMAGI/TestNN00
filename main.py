import test00 as test_lib
import os
import time
import utils as utils_lib
from log import Log

log_lib = Log.get_instance()
Log.set_min_level(0)

program_name = "TestNN00"

def main():
    log_lib.info("Starting %s ..." % (program_name))
    start_time = time.time()

    #test_lib.GettingStartedTF2()
    test_lib.ImageClassification()

    elapsed_time = time.time() - start_time
    log_lib.info("%s Completed in %s" % (program_name, utils_lib.elapsed_time_string(elapsed_time)))

if __name__ == "__main__":
    main()
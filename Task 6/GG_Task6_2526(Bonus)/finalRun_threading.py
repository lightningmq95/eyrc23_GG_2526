"""
* Team Id: GG_2526
* Author List: Harshal Kale, Akash Mohapatra, Sharanya Anil
* Filename: finalRun_original.py
* Theme: GeoGuide
* Functions: run_script, main.
* Global Variables: ip
"""

import threading
import subprocess
import time

"""
* Function Name: run_script
* Input: script_name(string) - THe name of the script to be executed.
* Output: None
* Logic: Executes the given python script.
* Example Call: run_script(script_name)
"""

def run_script(script_name):
    subprocess.run(["python", script_name])

"""
* Function Name: main
* Input: None
* Output: None
* Logic: Executes two scripts concurrently in separate threads, with a delay of 40 seconds between their executions.
* Example Call: main()
"""
if __name__ == "__main__":
    script1_thread = threading.Thread(target=run_script, args=("qgis.py",))
    script1_thread.start()

    time.sleep(40)  # Wait for 40 seconds

    script2_thread = threading.Thread(target=run_script, args=("finalRun_bonus.py",))
    script2_thread.start()

    script1_thread.join()
    script2_thread.join()

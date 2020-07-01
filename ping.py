import subprocess
import platform
import random

latency = []
jitter = []
jitter_average = 0
 
def ping_ip(current_ip_address):
    latency = []
    jitter = []
    try:
        output = subprocess.check_output("ping -{} 5 {}".format('n' if platform.system().lower(
        ) == "windows" else 'c', current_ip_address ), shell=True, universal_newlines=True)
        
        for response in output.splitlines():
            pos_of_time = response.find("time=")
            if pos_of_time != -1:
                latency.append(float(response[pos_of_time + 5 : pos_of_time + 10]))
        for i in range(0, len(latency)-1):
            jitter.append(abs(latency[i] - latency[i+1]))
        
        latency.pop(0)
        return latency, jitter
    except Exception:
            return None, None
 
if __name__ == '__main__':
    ip = '172.16.4.63'
    packet_size = random.randint(56, 248) # + 8 Byte icmp header
    latency, jitter = ping_ip(ip)
    jitter_average = sum(jitter)/len(jitter)
    print(f"{ip} {packet_size} Byte")
    print(f"latency = {latency}")
    print(f"Jitter = {jitter}")
    print(f"Average Jitter = {jitter_average}")
    print("-----------------------------")

    
import subprocess
import os


#将三种协议对应的zmap扫描命令集成在一个探测脚本中
# -------------------------- 配置参数 --------------------------
SRC_IPV6 = "xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx"  # IPv6地址
TARGET_FILE = "candidate.csv"                       # 候选地址文件
RATE = 10000                                       # 扫描速率（pps）
OUTPUT_DIR = "./"                                  # 结果输出目录
# --------------------------------------------------------------

def run_zmap_scan(protocol_name, scan_module, port=None):
    """
    运行一次ZMap扫描
    """
    output_file = os.path.join(OUTPUT_DIR, f"active_{protocol_name}.csv")
    cmd = [
        "sudo", "zmap",
        "--ipv6-source-ip", SRC_IPV6,
        "--ipv6-target-file", TARGET_FILE,
        "-M", scan_module,
        "-r", str(RATE),
        "-o", output_file
    ]
    if port is not None:
        cmd.extend(["-p", str(port)])

    print(f"\n[+] 开始 {protocol_name} 扫描，命令：")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] {protocol_name} 扫描完成，结果保存至：{output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"[✗] {protocol_name} 扫描失败：{e}")
        return None

def merge_results(file_list):
    """
    合并多个扫描结果，去重后输出总活跃地址
    """
    all_ips = set()
    for f in file_list:
        if f and os.path.exists(f):
            with open(f, "r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        all_ips.add(line)

    merged_file = os.path.join(OUTPUT_DIR, "active_all.csv")
    with open(merged_file, "w", encoding="utf-8") as outfile:
        for ip in sorted(all_ips):
            outfile.write(ip + "\n")

    print(f"\n[✓] 所有扫描完成，共发现 {len(all_ips)} 个活跃地址")
    print(f"[✓] 合并结果已保存至：{merged_file}")

if __name__ == "__main__":
    print("===== IPv6 多协议并行探测脚本 =====")
    print(f"源IPv6地址：{SRC_IPV6}")
    print(f"候选地址文件：{TARGET_FILE}")
    print(f"扫描速率：{RATE} pps")

    # 1. ICMPv6 扫描
    icmp_file = run_zmap_scan("icmp", "icmp6_echoscan")

    # 2. TCP/80 扫描
    tcp80_file = run_zmap_scan("tcp80", "tcp_synscan", port=80)

    # 3. TCP/443 扫描
    tcp443_file = run_zmap_scan("tcp443", "tcp_synscan", port=443)

    # 合并结果
    merge_results([icmp_file, tcp80_file, tcp443_file])
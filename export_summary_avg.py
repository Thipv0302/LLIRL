import os
import xml.etree.ElementTree as ET
from collections import defaultdict

FOLDER = "result_test/m"
OUT = "result_test/m/summary_avg.xml"

# Tìm tất cả file summary
files = sorted([
    os.path.join(FOLDER, f)
    for f in os.listdir(FOLDER)
    if f.startswith("summary_ep_") and f.endswith(".xml")
])

print("Found", len(files), "files")

all_steps = set()
data = defaultdict(lambda: defaultdict(list))

# Lấy toàn bộ step từ tất cả file
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    for step in root.findall("step"):
        all_steps.add(float(step.get("time")))

# Gom dữ liệu theo step
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()

    file_steps = {float(s.get("time")): s for s in root.findall("step")}

    for t in all_steps:
        if t in file_steps:
            step = file_steps[t]
            for attr, val in step.attrib.items():
                if attr != "time":
                    data[t][attr].append(float(val))

# Tạo XML output
root_out = ET.Element("summary")

for t in sorted(all_steps):
    s = ET.SubElement(root_out, "step")
    s.set("time", f"{t:.2f}")

    for attr, vals in data[t].items():
        avg_val = sum(vals) / len(vals)
        s.set(attr, f"{avg_val:.4f}")

# -------------------------
# Hàm format XML đẹp (xuống dòng từng step)
# -------------------------
def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

indent(root_out)

# Ghi file
tree = ET.ElementTree(root_out)
tree.write(OUT, encoding="utf-8", xml_declaration=True)

print("DONE! Saved:", OUT)

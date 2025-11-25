import xml.etree.ElementTree as ET
import sys

def count_vehicles(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        total_vehicles = 0
        
        # Count explicit vehicles
        vehicles = root.findall('vehicle')
        total_vehicles += len(vehicles)
        print(f"Explicit vehicles: {len(vehicles)}")
        
        # Count trips
        trips = root.findall('trip')
        total_vehicles += len(trips)
        print(f"Trips: {len(trips)}")
        
        # Calculate flows
        flows = root.findall('flow')
        flow_vehicles = 0
        for flow in flows:
            begin = float(flow.get('begin'))
            end = float(flow.get('end'))
            duration = end - begin
            
            if flow.get('number'):
                count = float(flow.get('number'))
                flow_vehicles += count
            elif flow.get('probability'):
                prob = float(flow.get('probability'))
                count = duration * prob
                flow_vehicles += count
            elif flow.get('period'):
                period = float(flow.get('period'))
                count = duration / period
                flow_vehicles += count
            elif flow.get('vehsPerHour'):
                vph = float(flow.get('vehsPerHour'))
                count = duration * vph / 3600
                flow_vehicles += count
            else:
                print(f"Warning: Unknown flow definition for id {flow.get('id')}")
                
        print(f"Flow vehicles (expected): {flow_vehicles:.2f}")
        total_vehicles += flow_vehicles
        
        print(f"Total expected vehicles: {total_vehicles:.2f}")
        
    except Exception as e:
        print(f"Error parsing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        count_vehicles(sys.argv[1])
    else:
        print("Usage: python count_vehicles.py <route_file>")

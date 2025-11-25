"""
Script to generate SUMO route file with target number of vehicles
by scaling probabilities from existing route file
"""
import xml.etree.ElementTree as ET
import sys
import os

def generate_route(input_file, output_file, target_vehicles=10000):
    """
    Generate route file with target number of vehicles by scaling probabilities
    
    Args:
        input_file: Path to input route file
        output_file: Path to output route file
        target_vehicles: Target number of vehicles
    """
    # Parse input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Calculate current expected vehicles
    current_vehicles = 0
    flows = root.findall('flow')
    
    print(f"Found {len(flows)} flow definitions")
    
    for flow in flows:
        begin = float(flow.get('begin'))
        end = float(flow.get('end'))
        duration = end - begin
        
        if flow.get('probability'):
            prob = float(flow.get('probability'))
            count = duration * prob
            current_vehicles += count
    
    print(f"\nCurrent expected vehicles: {current_vehicles:.2f}")
    print(f"Target vehicles: {target_vehicles}")
    
    # Calculate scale factor
    scale_factor = target_vehicles / current_vehicles
    print(f"Scale factor: {scale_factor:.4f}")
    
    # Scale all probabilities
    scaled_count = 0
    for flow in flows:
        if flow.get('probability'):
            old_prob = float(flow.get('probability'))
            new_prob = old_prob * scale_factor
            flow.set('probability', str(new_prob))
            
            # Verify
            begin = float(flow.get('begin'))
            end = float(flow.get('end'))
            duration = end - begin
            scaled_count += duration * new_prob
    
    # Save new file
    # Add XML declaration manually
    with open(output_file, 'wb') as f:
        f.write(b"<?xml version='1.0' encoding='utf-8'?>\n")
        tree.write(f, encoding='utf-8')
    
    print(f"\nGenerated route file: {output_file}")
    print(f"Verified expected vehicles: {scaled_count:.2f}")
    
    return True

if __name__ == "__main__":
    # Default paths
    input_file = "route_morning_6to10.rou.xml"
    output_file = "route_morning_6to10_10k.rou.xml"
    target_vehicles = 10000
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target_vehicles = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if len(sys.argv) > 3:
        input_file = sys.argv[3]
    
    print(f"Generating route file with {target_vehicles} vehicles...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}\n")
    
    success = generate_route(input_file, output_file, target_vehicles)
    
    if success:
        print("\n[SUCCESS] Route file generated successfully!")
    else:
        print("\n[ERROR] Failed to generate route file")
        sys.exit(1)


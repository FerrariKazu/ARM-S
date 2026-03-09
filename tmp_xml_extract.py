import xml.etree.ElementTree as ET
tree = ET.parse('models/mjcf/srl_robot.xml')
root = tree.getroot()
for body in root.iter('body'):
    if body.get('name') == 'torso_base':
        ET.dump(body)
        break

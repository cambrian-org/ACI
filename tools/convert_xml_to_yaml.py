"""This script will convert an xml file to a yaml file as follows:

<parent_key>
    <child_key attribute_key="attribute_value">
        <subchild_key attribute_key="attribute_value" attribute_key="attribute_value"/>
        <subchild_key>
            <subsubchild_key attribute_key="attribute_value"/>
        </subchild_key>
    </child_key>
</parent_key>

This would create the following xml:

parent_key:
    - child_key:
        - attribute_key: attribute_value
        - subchild_key:
            - attribute_key: attribute_value
            - attribute_key: attribute_value
        - subchild_key:
            - subsubchild_key: attribute_value

# NOTE: Child keys are always lists. If the child key only has one value (i.e. it's a string) that represents an attribute. If it's a list, it is added recursively.

"""

if __name__ == "__main__":
    import argparse
    import xml.etree.ElementTree as ET
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Converts an xml file to a yaml file")
    parser.add_argument("input_xml_file", type=str, help="The input xml file")
    parser.add_argument("output_yaml_file", type=str, help="The output yaml file")
    args = parser.parse_args()

    tree = ET.parse(args.input_xml_file)
    root = tree.getroot()

    def parse_element(element: ET.Element, result: list) -> list:
        # Set's attributes
        for key, value in element.items():
            result.append({key: value})

        # Adds children
        for child in element:
            result.append({child.tag: parse_element(child, [])})
        return result

    yaml_out = [dict(mujoco=parse_element(root, []))]
    with open(args.output_yaml_file, "w") as f:
        f.write("# This file was generated by tools/convert_xml_to_yaml.py\n")
        yaml.dump(yaml_out, f)
    with open(Path(args.output_yaml_file).with_suffix(".xml"), "w") as f:
        f.write("<!-- This file was generated by tools/convert_xml_to_yaml.py -->\n")
        f.write(ET.tostring(root, encoding="unicode"))

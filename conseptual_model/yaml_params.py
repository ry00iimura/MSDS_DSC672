import yaml
import os

class YamlManager:
    def __init__(self, filename):
        self.filename = filename

    def write_yaml(self, data):
        """YAMLファイルへの書き出し"""
        with open(self.filename, 'w') as file:
            yaml.dump(data, file)
    
    def append_yaml(self, data):
        """YAMLファイルへの追記"""
        existing_data = self.read_yaml()
        existing_data.update(data)
        self.write_yaml(existing_data)
    
    def update_yaml(self, key, value):
        """YAMLファイルの内容を修正"""
        data = self.read_yaml()
        data[key] = value
        self.write_yaml(data)
    
    def delete_from_yaml(self, key):
        """YAMLファイルから特定のキーを削除"""
        data = self.read_yaml()
        if key in data:
            del data[key]
            self.write_yaml(data)
    
    def read_yaml(self):
        """YAMLファイルの読み込み"""
        if not os.path.exists(self.filename):
            return {}
        with open(self.filename, 'r') as file:
            return yaml.safe_load(file) or {}

# # 使用例
if __name__ == '__main__':
    yaml_manager = YamlManager('sample.yaml')
    data = {'key1': 'value1', 'key2': 'value2'}
    yaml_manager.write_yaml(data)
    yaml_manager.append_yaml({'key3': 'value3'})
    yaml_manager.update_yaml('key1', 'new_value1')
    yaml_manager.delete_from_yaml('key2')
    loaded_data = yaml_manager.read_yaml()
    print(loaded_data)

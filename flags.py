from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import json


class Flags(dict):
  """For storing and accessing flags."""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def print_values(self, indent=1):
    for k, v in sorted(self.items()):
      if isinstance(v, Flags):
        print('{}{}:'.format('\t'*indent, k))
        v.print_values(indent=indent+1)
      else:
        print('{}{}: {}'.format('\t'*indent, k, v))

  def load(self, other):
    def recursive_update(flags, source_dict):
      for k in source_dict:
        if isinstance(source_dict[k], dict):
          flags[k] = Flags()
          recursive_update(flags[k], source_dict[k])
        else:
          flags[k] = source_dict[k]
    recursive_update(self, other)

  def load_json(self, json_string):
    other = json.loads(json_string)
    self.load(other)

  def load_b64json(self, json_string):
    json_string = base64.b64decode(json_string)
    print('!!!!!!JSON STRING!!!!!!!!')
    print(json_string)
    self.load_json(json_string)

  def set_if_empty(self, key, val):
    if key not in self:
      self[key] = val

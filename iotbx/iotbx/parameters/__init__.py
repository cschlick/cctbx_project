from iotbx import simple_tokenizer
from scitbx.python_utils.str_utils import line_breaker
from libtbx.itertbx import count
from libtbx import introspection
import os

standard_identifier_start_characters = {}
for c in "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
  standard_identifier_start_characters[c] = None
standard_identifier_continuation_characters = dict(
  standard_identifier_start_characters)
for c in ".0123456789":
  standard_identifier_continuation_characters[c] = None

def is_standard_identifier(string):
  if (len(string) == 0): return False
  if (string[0] not in standard_identifier_start_characters): return False
  for c in string[1:]:
    if (c not in standard_identifier_continuation_characters): return False
  sub_strings = string.split(".")
  if (len(sub_strings) > 1):
    for sub in sub_strings:
      if (not is_standard_identifier(sub)): return False
  return True

def str_from_value_words(value_words):
  if (len(value_words) == 1 and value_words[0].value.lower() == "none"):
    return None
  return " ".join([word.value for word in value_words])

def bool_from_value_words(value_words):
  value_string = str_from_value_words(value_words)
  if (value_string is None): return None
  word_lower = value_words[0].value.lower()
  if (word_lower == "none"): return None
  if (word_lower in ["false", "no", "off", "0"]): return False
  if (word_lower in ["true", "yes", "on", "1"]): return True
  assert len(value_words) > 0
  raise RuntimeError(
    'One True of False value expected, "%s" found%s' % (
      value_string, value_words[0].where_str()))

def number_from_value_words(value_words, number_type, number_type_name):
  value_string = str_from_value_words(value_words)
  if (value_string is None): return None
  try: return number_type(value_string)
  except:
    raise RuntimeError(
      '%s value expected, "%s" found%s' % (
        number_type_name, value_string, value_words[0].where_str()))

def int_from_value_words(value_words):
  return number_from_value_words(value_words, int, "Integer")

def float_from_value_words(value_words):
  return number_from_value_words(value_words, float, "Floating-point")

default_definition_type_names = [
  "bool", "int", "float", "str",
  "choice", "multi_choice",
  "path", "key",
  "unit_cell", "space_group"]

def definition_type_from_value_words(value_words, type_names):
  if (len(value_words) == 1):
    word_lower = value_words[0].value.lower()
    if (word_lower == "none"): return None
    if (word_lower in ["bool", "int", "float", "str",
                       "choice", "multi_choice",
                       "path", "key"]):
      return word_lower
  assert len(value_words) > 0
  raise RuntimeError(
    'Unexpected definition type: "%s"%s' % (
      value_words[0].value, value_words[0].where_str()))

def show_attributes(self, out, prefix, attributes_level, print_width):
  if (attributes_level <= 0): return
  for name in self.attribute_names:
    value = getattr(self, name)
    if ((name == "help" and value is not None)
        or (value is not None and attributes_level > 1)
        or attributes_level > 2):
      if (not isinstance(value, str)):
        print >> out, prefix+"  ."+name, value
      else:
        value = str(simple_tokenizer.word(value=value, quote_token='"'))
        indent = " " * (len(prefix) + 3 + len(name) + 1)
        if (len(indent+value) < print_width):
          print >> out, prefix+"  ."+name, value
        else:
          is_first = True
          for block in line_breaker(value[1:-1], print_width-2-len(indent)):
            if (is_first):
              print >> out, prefix+"  ."+name, '"'+block+'"'
              is_first = False
            else:
              print >> out, indent+'"'+block+'"'

class definition:

  def __init__(self,
        name,
        values,
        help=None,
        caption=None,
        short_caption=None,
        required=None,
        type=None,
        input_size=None,
        expert_level=None):
    introspection.adopt_init_args()
    self.attribute_names = self.__init__varnames__[3:]

  def copy(self, values):
    keyword_args = {}
    for keyword in self.__init__varnames__[1:]:
      keyword_args[keyword] = getattr(self, keyword)
    keyword_args["values"] = values
    return definition(**keyword_args)

  def has_attribute_with_name(self, name):
    return name in self.attribute_names

  def assign_attribute(self, name, value_words, type_names):
    assert self.has_attribute_with_name(name)
    if (name == "required"):
      value = bool_from_value_words(value_words)
    elif (name == "type"):
      value = definition_type_from_value_words(value_words, type_names)
    elif (name in ["input_size", "expert_level"]):
      value = int_from_value_words(value_words)
    else:
      value = str_from_value_words(value_words)
    setattr(self, name, value)

  def show(self, out, prefix="", attributes_level=0, print_width=79,
                 previous_object=None):
    if (previous_object is not None
        and not isinstance(previous_object, definition)):
      print >> out, prefix.rstrip()
    line = prefix+self.name
    indent = " " * len(line)
    for word in self.values:
      line_plus = line + " " + str(word)
      if (len(line_plus) > print_width-2 and len(line) > len(indent)):
        print >> out, line + " \\"
        line = indent + " " + str(word)
      else:
        line = line_plus
    print >> out, line
    show_attributes(
      self=self,
      out=out,
      prefix=prefix,
      attributes_level=attributes_level,
      print_width=print_width)

  def all_definitions(self):
    return [self]

  def get(self, path):
    if (self.name == path): return [self]
    return []

  def automatic_type(self):
    types = {}
    for word in self.values:
      if (word.quote_token is not None):
        types["str"] = None
        continue
      word_lower = word.value.lower()
      if (word_lower in ["false", "no", "off"
                         "true", "yes", "on"]):
        types["bool"] = None
        continue
      try: py_value = eval(word.value, {}, {})
      except:
        if (word.value[0] in standard_identifier_start_characters):
          types["str"] = None
        else:
          types["unknown"] = None
        continue
      if (isinstance(py_value, float)):
        types["float"] = None
        continue
      if (isinstance(py_value, int)):
        types["int"] = None
        continue
    types = types.keys()
    types.sort()
    if (types == ["int", "float"]): return "float"
    if (len(types) == 1): return types[0]
    return None

  def automatic_type_assignment(self, assignment_if_unknown=None):
    if (self.type is None):
      self.type = self.automatic_type()
      if (self.type is None):
        self.type = assignment_if_unknown

class scope:

  def __init__(self,
        name,
        objects):
    if (objects is None): objects = []
    introspection.adopt_init_args()

  def show(self, out, prefix="", attributes_level=0, print_width=79,
                 previous_object=None):
    if (previous_object is not None):
      print >> out, prefix.rstrip()
    print >> out, prefix + self.name + " {"
    previous_object = None
    for object in self.objects:
      object.show(
        out=out,
        prefix=prefix+"  ",
        attributes_level=attributes_level,
        print_width=print_width,
        previous_object=previous_object)
      previous_object = object
    print >> out, prefix + "}"

  def all_definitions(self):
    result = []
    for object in self.objects:
      result.extend(object.all_definitions())
    return result

  def get(self, path):
    if (self.name == path): return [self]
    if (not path.startswith(self.name+".")): return []
    path = path[len(self.name)+1:]
    result = []
    for object in self.objects:
      result.extend(object.get(path=path))
    return result

class table:

  def __init__(self,
        name,
        row_names,
        row_objects,
        style=None,
        help=None,
        caption=None,
        short_caption=None,
        sequential_format=None,
        disable_add=None,
        disable_delete=None,
        expert_level=None):
    introspection.adopt_init_args()
    self.attribute_names = self.__init__varnames__[4:]
    assert style in [None, "row", "column", "block", "page"]
    if (sequential_format is not None):
      assert isinstance(sequential_format % 0, str)

  def has_attribute_with_name(self, name):
    return name in self.attribute_names

  def assign_attribute(self, name, value_words):
    assert self.has_attribute_with_name(name)
    if (name in ["disable_add", "disable_delete"]):
      value = bool_from_value_words(value_words)
    elif (name in ["expert_level"]):
      value = int_from_value_words(value_words)
    else:
      value = str_from_value_words(value_words)
      if (name == "style"):
        style = value
        assert style in [None, "row", "column", "block", "page"]
      elif (name == "sequential_format"):
        sequential_format = value
        if (sequential_format is not None):
          assert isinstance(sequential_format % 0, str)
    setattr(self, name, value)

  def add_row(self, name, objects):
    self.row_names.append(name)
    self.row_objects.append(objects)

  def show(self, out, prefix="", attributes_level=0, print_width=79,
                 previous_object=None):
    if (previous_object is not None):
      print >> out, prefix.rstrip()
    print >> out, "%stable %s" % (prefix, self.name)
    show_attributes(
      self=self,
      out=out,
      prefix=prefix,
      attributes_level=attributes_level,
      print_width=print_width)
    print >> out, prefix+"{"
    assert len(self.row_names) == len(self.row_objects)
    for name,objects in zip(self.row_names, self.row_objects):
      s = prefix + "  "
      if (name is not None):
        s += name + " "
      print >> out, s+"{"
      previous_object = None
      for object in objects:
        object.show(
          out=out,
          prefix=prefix+"    ",
          attributes_level=attributes_level,
          print_width=print_width,
          previous_object=previous_object)
        previous_object = object
      print >> out, prefix+"  }"
    print >> out, prefix+"}"

  def all_definitions(self):
    result = []
    for row_object in self.row_objects:
      for object in row_object:
        result.extend(object.all_definitions())
    return result

  def get(self, path):
    if (self.name == path): return [self]
    if (not path.startswith(self.name+".")): return []
    path = path[len(self.name)+1:]
    result = []
    for n_row,row_name,row_objects in zip(count(1),
                                          self.row_names,
                                          self.row_objects):
      for alt_row_name in [row_name, str(n_row)]:
        if (alt_row_name is None): continue
        if (alt_row_name == path):
          result.extend(row_objects)
        elif (path.startswith(alt_row_name+".")):
          for row_object in row_objects:
            result.extend(row_object.get(path=path[len(alt_row_name)+1:]))
    return result

class object_list:

  def __init__(self, objects):
    self.objects = objects

  def show(self, out, prefix="", attributes_level=0, print_width=None):
    if (print_width is None):
      print_width = 79
    previous_object = None
    for object in self.objects:
      object.show(
        out=out,
        prefix=prefix,
        attributes_level=attributes_level,
        print_width=print_width,
        previous_object=previous_object)
      previous_object = object
    return self

  def all_definitions(self):
    result = []
    for object in self.objects:
      result.extend(object.all_definitions())
    return result

  def get(self, path):
    result = []
    for object in self.objects:
      result.extend(object.get(path))
    return object_list(objects=result)

  def get_with_variable_substitution(self, path, path_memory=None):
    if (path_memory is None):
      path_memory = {path: None}
    elif (path not in path_memory):
      path_memory[path] = None
    else:
      raise RuntimeError("Dependency cycle in variable substitution: $%s" % (
        path))
    result_raw = self.get(path=path)
    result_sub = []
    for object in result_raw.objects:
      if (not isinstance(object, definition)):
        result_sub.append(object)
      else:
        result_sub.append(self.variable_substitution(
          object=object, path_memory=path_memory))
    del path_memory[path]
    return object_list(objects=result_sub)

  def variable_substitution(self, object, path_memory):
    new_values = []
    for word in object.values:
      if (word.quote_token == "'"): continue
      substitution_proxy = variable_substitution_proxy(word)
      for fragment in substitution_proxy.fragments:
        if (not fragment.is_variable):
          fragment.result = simple_tokenizer.word(
            value=fragment.value, quote_token='"')
          continue
        variable_values = None
        for variable_object in self.get_with_variable_substitution(
                                 path=fragment.value,
                                 path_memory=path_memory).objects:
          if (isinstance(variable_object, definition)):
            variable_values = variable_object.values
        if (variable_values is None):
          env_var = os.environ.get(fragment.value, None)
          if (env_var is not None):
            variable_values = [simple_tokenizer.word(
              value=env_var,
              quote_token='"')]
        if (variable_values is None):
          raise RuntimeError("Undefined variable: $%s%s" % (
            fragment.value, word.where_str()))
        if (not substitution_proxy.force_string):
          fragment.result = variable_values
        else:
          fragment.result = simple_tokenizer.word(
            value=" ".join([str(v) for v in variable_values]),
            quote_token='"')
      new_values.extend(substitution_proxy.get_new_values())
    return object.copy(values=new_values)

  def process_includes(self,
        definition_type_names,
        reference_directory,
        include_memory=None):
    if (definition_type_names is None):
      definition_type_names = default_definition_type_names
    if (include_memory is None): include_memory = {}
    result = []
    for object in self.objects:
      if (not isinstance(object, definition)
          or object.name != "include"):
        result.append(object)
      else:
        object_sub = self.variable_substitution(object=object, path_memory={})
        for file_name in [word.value for word in object_sub.values]:
          if (reference_directory is not None
              and not os.path.isabs(file_name)):
            file_name = os.path.join(reference_directory, file_name)
          file_name_normalized = os.path.normpath(os.path.abspath(file_name))
          if (file_name_normalized in include_memory): continue
          include_memory[file_name_normalized] = None
          result.extend(parse(
            file_name=file_name,
            definition_type_names=definition_type_names).process_includes(
              definition_type_names=definition_type_names,
              reference_directory=os.path.dirname(file_name_normalized),
              include_memory=include_memory).objects)
    return object_list(objects=result)

  def automatic_type_assignment(self, assignment_if_unknown=None):
    for object in self.all_definitions():
      object.automatic_type_assignment(
        assignment_if_unknown=assignment_if_unknown)

class variable_substitution_fragment:

  def __init__(self, is_variable, value):
    introspection.adopt_init_args()

class variable_substitution_proxy:

  def __init__(self, word):
    self.word = word
    self.force_string = word.quote_token is not None
    self.have_variables = False
    self.fragments = []
    fragment_value = ""
    char_iter = simple_tokenizer.character_iterator(word.value)
    c = char_iter.next()
    while (c is not None):
      if (c != "$"):
        fragment_value += c
        if (c == "\\" and char_iter.look_ahead() == "$"):
          fragment_value += char_iter.next()
        c = char_iter.next()
      else:
        self.have_variables = True
        if (len(fragment_value) > 0):
          self.fragments.append(variable_substitution_fragment(
            is_variable=False,
            value=fragment_value))
          fragment_value = ""
        c = char_iter.next()
        if (c is None):
          word.raise_syntax_error("$ must be followed by an identifier: ")
        if (c == "{"):
          self.force_string = True
          while True:
            c = char_iter.next()
            if (c is None):
              word.raise_syntax_error('missing "}": ')
            if (c == "}"):
              c = char_iter.next()
              break
            fragment_value += c
          if (not is_standard_identifier(fragment_value)):
            word.raise_syntax_error("improper variable name ")
          self.fragments.append(variable_substitution_fragment(
            is_variable=True,
            value=fragment_value))
        else:
          if (c not in standard_identifier_start_characters):
            word.raise_syntax_error("improper variable name ")
          fragment_value = c
          while True:
            c = char_iter.next()
            if (c is None): break
            if (c == "."): break
            if (c not in standard_identifier_continuation_characters): break
            fragment_value += c
          self.fragments.append(variable_substitution_fragment(
            is_variable=True,
            value=fragment_value))
        fragment_value = ""
    if (len(fragment_value) > 0):
      self.fragments.append(variable_substitution_fragment(
        is_variable=False,
        value=fragment_value))
    if (len(self.fragments) > 1):
      self.force_string = True

  def get_new_values(self):
    if (not self.have_variables):
      return [self.word]
    if (not self.force_string):
      return self.fragments[0].result
    return [simple_tokenizer.word(
      value="".join([fragment.result.value for fragment in self.fragments]),
      quote_token='"')]

def parse(
      input_string=None,
      file_name=None,
      definition_type_names=None,
      process_includes=False):
  from iotbx.parameters import parser
  if (input_string is None):
    assert file_name is not None
    input_string = open(file_name).read()
  if (definition_type_names is None):
    definition_type_names = default_definition_type_names
  result = object_list(objects=parser.collect_objects(
    word_stack=simple_tokenizer.as_word_stack(
      input_string=input_string,
      file_name=file_name,
      contiguous_word_characters="",
      auto_split_unquoted={"{}": ("{", "}")}),
    definition_type_names=definition_type_names))
  if (process_includes):
    if (file_name is None):
      reference_directory = None
    else:
      reference_directory = os.path.dirname(os.path.abspath(file_name))
    result = result.process_includes(
      definition_type_names=definition_type_names,
      reference_directory=reference_directory)
  return result

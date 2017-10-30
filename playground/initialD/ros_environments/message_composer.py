"""Remote ROS Message.

has to types
RMsg1 = (class_name, built-in) -> Msg1 = class(builtin_value)
RMsg2 = (class_name, {'field1': RMsg21, 'field2': RMsg22}) ->
"""
import importlib

__type_mapping__ = {
    # bool
    'bool': 'bool',
    # int
    'int': 'int', 'char': 'int', 'uint32': 'int', 'int8': 'int', 'int16': 'int', 'int32': 'int',
    # float
    'float16': 'float', 'float32': 'float', 'float64': 'float',
    # string
    'string': 'str',
    # time
    'time': 'genpy.Time'
}

def import_msg_cls(msg_name):
    """Import message class according to message name.

    Import the actual ROS message class according to message name using
    `importlib`. Also handles builtin classes.
    """
    try:
        msg_name_split = msg_name.split('.')
        if len(msg_name_split) == 1:  # builtin classes
            package_name = '__builtin__'
        else:
            package_name = '.'.join(msg_name_split[:-1])
        class_name =  msg_name_split[-1]
        msg_cls = getattr(
            importlib.import_module(package_name), class_name
        )
    except:
        raise ImportError(
            '[message_composer.import_cls()]: message class '
            + msg_name + ' not found.'
        )

    return msg_cls

def parse_msg_type(msg_type):
    """Parse message type string to use with ` import_msg_cls`.

    Parse the string of message class found in the `_slot_types` attribute.
    Take special care to list type slots, append `[]` to the message class so
    MetaMessageComposer knows to make the right form of `__call__()` function.
    """
    # Built in types
    if msg_type in __type_mapping__:
        return __type_mapping__[msg_type]
    # List type message
    elif '[]' in msg_type:
        msg_type = msg_type[:-2]  # strip trailing `[` and `]`
        return '[]' + parse_msg_type(msg_type)
    # Normal message
    else:
        msg_type_split = msg_type.split('/')
        if len(msg_type_split) == 2:
            return msg_type_split[0] + '.msg.' + msg_type_split[1]
        else:
            raise ValueError(
                'message_composer.parse_msg_type: do not recognize message'
                ' type {}'.format(msg_type))

def single_call(self, data):
    kwargs = {}
    # raise default
    if data is None:
        return self.cls()
    else:
        # keyword argument form
        if type(data) is dict:
            for slot in data:
                kwargs[slot] = self._sub_composers[slot](data[slot])
        # positional argument form
        elif type(data) is tuple or type(data) is list:
            assert len(data) == len(self._slots)
            for i, slot in self._slots:
                kwargs[slot] = self._sub_composer[slot](data[i])
        # single-field form
        else:
            try:
                assert len(self._slots) == 1 and self._slots[0] == 'data'
                kwargs['data'] = self._sub_composers['data'](data)
            except:
                raise ValueError(
                    'Message data not in correct form.'
                )

        return self.cls(**kwargs)

def list_call(self, data):
    return map(lambda x: single_call(self, x), data)

class MetaMessageComposer(type):
    """Meta class for message composers."""
    __composers = {}
    def __new__(cls, msg_name):
        # make sure composer is built only once, a.k.a singleton
        if msg_name in MetaMessageComposer.__composers:
            return MetaMessageComposer.__composers[msg_name]

        # import message class
        if '[]' in msg_name:
            msg_cls = import_msg_cls(msg_name[2:])
        else:
            msg_cls = import_msg_cls(msg_name)

        # build MessageComposer class for imported message
        # Recursively build composers for slots.
        name = 'MessageComposer_' + msg_name
        try:
            _slots = getattr(msg_cls, '__slots__')
            _slot_types = getattr(msg_cls, '_slot_types')
            _slot_msg_names = map(parse_msg_type, _slot_types)
            _sub_composers = {}
            for i, slot in enumerate(_slots):
                _sub_composers[slot] = MetaMessageComposer(_slot_msg_names[i])
            attr = {
                '_slots': _slots,
                '_sub_composers': _sub_composers,
                '__call__': list_call if '[]' in msg_name else single_call,
            }
        except AttributeError:
            attr = {'__call__': msg_cls.__call__}

        attr['cls'] = msg_cls
        composer_cls = super(MetaMessageComposer, cls).__new__(
            cls, name, (object,), attr)
        composer = composer_cls()
        MetaMessageComposer.__composers[msg_name] = composer

        return composer

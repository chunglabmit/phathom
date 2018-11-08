def attach_callback(func, callback_name):
    def notify(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if callback_name == 'on_size_change':
                self.on_size_change(self)
            elif callback_name == 'on_item_change':
                self.on_item_change(self, args[0])
            else:
                raise ValueError("Unknown callback name: {0}".format(callback_name))
    return notify


class CallbackList(list):

    def __init__(self, *args, **kwargs):
        super(CallbackList, self).__init__(*args, **kwargs)
        self.on_size_change = lambda x: None
        self.on_item_change = lambda x, y: None

    extend = attach_callback(list.extend, 'on_size_change')
    append = attach_callback(list.append, 'on_size_change')
    remove = attach_callback(list.remove, 'on_size_change')
    insert = attach_callback(list.insert, 'on_size_change')
    pop = attach_callback(list.pop, 'on_size_change')
    __delitem__ = attach_callback(list.__delitem__, 'on_size_change')
    __setitem__ = attach_callback(list.__setitem__, 'on_item_change')

from callback_list import CallbackList


def test_callback_list_nocallbacks():

    # CHeck that things work properly without callbacks

    l = CallbackList([1, 2, 3])
    l.append(2)
    l.append(6)
    l.pop(3)
    l.remove(1)
    l.insert(0, 9)
    l.extend([4, 5, 4])
    l[2] = 9
    del l[1]

    assert l == [9, 9, 6, 4, 5, 4]


def test_callback_list():

    class Callbacks(object):

        def __init__(self):
            self.sizes = []
            self.indices = []

        def on_size_change(self, list_instance):
            self.sizes.append(len(list_instance))

        def on_item_change(self, list_instance, index):
            self.indices.append(index)

    callbacks = Callbacks()

    # CHeck that things work properly without callbacks

    l = CallbackList([1, 2, 3])

    # Register callbacks
    l.on_size_change = callbacks.on_size_change
    l.on_item_change = callbacks.on_item_change

    l.append(2)
    l.append(6)
    l.pop(3)
    l.remove(1)
    l.insert(0, 9)
    l.extend([4, 5, 4])
    l[2] = 9
    del l[1]

    assert l == [9, 9, 6, 4, 5, 4]
    assert callbacks.sizes == [4, 5, 4, 3, 4, 7, 6]
    assert callbacks.indices == [2]

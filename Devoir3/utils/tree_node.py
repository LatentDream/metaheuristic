
class TreeNode:
    def __init__(self, id) -> None:
        self.id = id
        self.parents = set()
        self.children = set()

    def add_children(self, *children):
        for child in children:
            self.children.add(child)
            child.parents.add(self)

    def add_parents(self, *parents):
        for parent in parents:
            self.parents.add(parent)
            parent.child.add(self)

    def depth(self):
        def find_depth(node):
            # Top node
            if len(node.parents) == 0:
                return 1
            # Middle node /!\ can have multiple parent
            depth = 0
            for parent in node.parents:
                 depth = max(depth, find_depth(parent))
            return depth + 1
        return find_depth(self)
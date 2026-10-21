"""Compare analysis structure independently of comments and display text."""
import ast
import hashlib
import json

ALIASES = {"load_transition_inputs": "load_week5_outputs", "save_graph_inputs": "save_week6_outputs"}

def is_display_call(call):
    f=call.func
    if isinstance(f,ast.Name):return f.id in {'print','ValueError','RuntimeError','FileNotFoundError','SystemExit','KeyError','AssertionError'}
    if isinstance(f,ast.Attribute):
        if f.attr in {'info','warning','error','debug','exception','critical','suptitle','set_title','set_xlabel','set_ylabel','set_zlabel'}:return True
        if f.attr=='append' and isinstance(f.value,ast.Attribute) and f.value.attr=='notes':return True
        if f.attr=='append' and isinstance(f.value,ast.Name) and f.value.id=='failures':return True
    return False

def canonical(value):
    if isinstance(value,ast.AST):return {'node':type(value).__name__,'fields':{k:canonical(v) for k,v in ast.iter_fields(value) if not(k=='type_params' and v==[])}}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,bytes):return {'literal':'bytes','hex':value.hex()}
    if isinstance(value,complex):return {'literal':'complex','real':value.real,'imag':value.imag}
    if value is Ellipsis:return {'literal':'ellipsis'}
    return value

class Scientific(ast.NodeTransformer):
    def __init__(self):self.function=None
    def _body(self,node):
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
        return node
    def visit_Module(self,node):return self.generic_visit(self._body(node))
    def visit_ClassDef(self,node):return self.generic_visit(self._body(node))
    def visit_FunctionDef(self,node):
        previous=self.function;self.function=node.name;node.name=ALIASES.get(node.name,node.name);result=self.generic_visit(self._body(node));self.function=previous;return result
    visit_AsyncFunctionDef=visit_FunctionDef
    def visit_Name(self,node):node.id=ALIASES.get(node.id,node.id);return node
    def display(self,node):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):node.value='<display text>'
        elif isinstance(node,ast.JoinedStr):
            for part in node.values:
                if isinstance(part,ast.Constant) and isinstance(part.value,str):part.value='<display text>'
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if is_display_call(node):node.args=[self.display(a) for a in node.args]
        if isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            for kw in node.keywords:
                if kw.arg in {'help','description'}:kw.value=self.display(kw.value)
        return node
    def visit_Assert(self,node):
        node=self.generic_visit(node)
        if node.msg is not None:node.msg=self.display(node.msg)
        return node
    def visit_Dict(self,node):
        node=self.generic_visit(node)
        if self.function in {'write_report','main'}:
            for key,value in zip(node.keys,node.values):
                descriptions = {'scope','remaining'} if self.function == 'write_report' else {'description','early_stopping_note'}
                if isinstance(key,ast.Constant) and key.value in descriptions:
                    if isinstance(value,ast.List):value.elts=[self.display(v) for v in value.elts]
                    else:self.display(value)
        return node

def science_hash(source):
    tree=Scientific().visit(ast.parse(source));return hashlib.sha256(json.dumps(canonical(tree),sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()

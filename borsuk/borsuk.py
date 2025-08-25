#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
BORSUK – prosty, własny język programowania (PL)
================================================

Jednoplikowy interpreter języka „Borsuk”.

Obsługa:
- typy: int, number (float), string, bool, null, array, map, instance
- zmienne: let x = 1; oraz przypisania złożone: x += 1;
- funkcje (domknięcia): func nazwa(a,b) { ... return a+b; }
- instrukcje: if/else, while, for-in, return, break, continue
- indeksowanie i wywołania: arr[0], map["k"], f(1,2)
- operatory: + - * / %  == != < <= > >=  && ||  !  - (unarny)
- klasy/obiekty: class C { func init(...) {...} func m(...) {...} }, new C(...), this.x
- wbudowane: print, len, range, type, keys, values, push, pop, str, num, join, split, map, filter, get, set
- import "plik.bk"
- interpolacja napisów: "x=${1+2}"

Uruchamianie:
$ python3 borsuk.py            # REPL
$ python3 borsuk.py program.bk # uruchom plik

'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import sys

############################################################
# LEXER
############################################################

TT_NUMBER = "NUMBER"
TT_STRING = "STRING"
TT_IDENT  = "IDENT"
TT_EOF    = "EOF"

KEYWORDS = {
    "let", "func", "if", "else", "while", "return", "true", "false", "null",
    "break", "continue", "for", "in", "import", "class", "new", "this"
}

@dataclass
class Token:
    type: str
    value: Any
    pos: int

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.n = len(text)

    def peek(self) -> str:
        return self.text[self.i] if self.i < self.n else "\0"

    def advance(self) -> str:
        ch = self.peek()
        self.i += 1
        return ch

    def skip_ws_and_comments(self):
        while True:
            while self.peek().isspace():
                self.advance()
            # // komentarz liniowy
            if self.peek() == '/' and self._peek2() == '/':
                while self.peek() not in ('\n', '\0'):
                    self.advance()
                continue
            # /* komentarz blokowy */
            if self.peek() == '/' and self._peek2() == '*':
                self.advance(); self.advance()
                while not (self.peek() == '*' and self._peek2() == '/'):
                    if self.peek() == '\0':
                        raise SyntaxError("Niedomknięty komentarz blokowy")
                    self.advance()
                self.advance(); self.advance()
                continue
            break

    def _peek2(self) -> str:
        j = self.i + 1
        return self.text[j] if j < self.n else "\0"

    def string(self) -> Token:
        quote = self.advance()  # consume ' or "
        start = self.i
        out = []
        while True:
            ch = self.peek()
            if ch == "\0":
                raise SyntaxError("Niedomknięty napis")
            if ch == '\\\\':
                self.advance()
                esc = self.advance()
                mapping = {'n':'\n','t':'\t','r':'\r','"':'"',"\'":"\'",'\\':'\\'}
                out.append(mapping.get(esc, esc))
                continue
            if ch == quote:
                self.advance()
                return Token(TT_STRING, ''.join(out), start)
            out.append(self.advance())

    def number(self) -> Token:
        start = self.i
        s = []
        dots = 0
        while self.peek().isdigit() or self.peek() == '.':
            if self.peek() == '.':
                dots += 1
                if dots > 1:
                    break
            s.append(self.advance())
        lex = ''.join(s)
        if '.' in lex:
            val = float(lex)
        else:
            val = int(lex)
        return Token(TT_NUMBER, val, start)

    def ident(self) -> Token:
        start = self.i
        s = []
        while self.peek().isalnum() or self.peek() == '_':
            s.append(self.advance())
        name = ''.join(s)
        if name in KEYWORDS:
            return Token(name.upper(), name, start)
        return Token(TT_IDENT, name, start)

    def two(self, a,b, t):
        if self.peek() == a and self._peek2() == b:
            self.advance(); self.advance()
            return Token(t, a+b, self.i-2)
        return None

    def next(self) -> Token:
        self.skip_ws_and_comments()
        ch = self.peek()
        if ch == '\0':
            return Token(TT_EOF, None, self.i)
        if ch in ('"', "'"):
            return self.string()
        if ch.isdigit():
            return self.number()
        if ch.isalpha() or ch == '_':
            return self.ident()

        # two-char operators (kolejność ma znaczenie)
        for (a,b,t) in [('+','=', 'PLUS_EQ'),('-','=', 'MINUS_EQ'),('*','=', 'STAR_EQ'),('/','=', 'SLASH_EQ'),('%','=', 'MOD_EQ'),
                        ('=','=', 'EQ'),('!','=', 'NE'),('<','=', 'LE'),('>','=', 'GE'),('&','&','AND'),('|','|','OR')]:
            tok = self.two(a,b,t)
            if tok: return tok

        # single-char tokens
        single = {
            '+':'PLUS','-':'MINUS','*':'STAR','/':'SLASH','%':'MOD',
            '(':'LP',')':'RP','{':'LB','}':'RB','[':'LBR',']':'RBR',
            ';':'SC','.' :'DOT', ',':'COMMA',':':'COLON', '=':'ASSIGN',
            '<':'LT','>':'GT','!':'BANG'
        }
        if ch in single:
            self.advance()
            return Token(single[ch], ch, self.i-1)
        raise SyntaxError(f"Nieznany znak: {ch!r} @ {self.i}")

############################################################
# PARSER – Pratt
############################################################

PREC = {
    'OR': 1,
    'AND': 2,
    'EQ': 3, 'NE': 3,
    'LT': 4, 'LE': 4, 'GT': 4, 'GE': 4,
    'PLUS': 5, 'MINUS': 5,
    'STAR': 6, 'SLASH': 6, 'MOD': 6,
}

class Parser:
    def __init__(self, lx: Lexer):
        self.lx = lx
        self.cur = lx.next()

    def eat(self, t: str) -> Token:
        if self.cur.type != t:
            raise SyntaxError(f"Spodziewano {t}, otrzymano {self.cur.type}")
        tok = self.cur
        self.cur = self.lx.next()
        return tok

    def match(self, *types) -> bool:
        if self.cur.type in types:
            self.cur = self.lx.next()
            return True
        return False

    # ------------- Wyrażenia -------------
    def parse_expr(self, prec=0):
        left = self.nud()
        while True:
            t = self.cur.type
            if t in PREC and PREC[t] >= prec:
                op = self.cur; self.eat(t)
                right = self.parse_expr(PREC[t] + 1)
                left = ('bin', op.type, left, right)
                continue
            if t == 'LP':
                args = self.parse_args()
                left = ('call', left, args)
                continue
            if t == 'LBR':
                self.eat('LBR')
                idx = self.parse_expr()
                self.eat('RBR')
                left = ('index', left, idx)
                continue
            if t == 'DOT':
                self.eat('DOT')
                name = self.eat(TT_IDENT).value
                left = ('getprop', left, name)
                continue
            break
        return left

    def nud(self):
        t = self.cur.type
        if t == TT_NUMBER:
            v = self.cur.value; self.eat(TT_NUMBER)
            return ('num', v)
        if t == TT_STRING:
            v = self.cur.value; self.eat(TT_STRING)
            return ('str', v)
        if t == 'TRUE': self.eat('TRUE'); return ('bool', True)
        if t == 'FALSE': self.eat('FALSE'); return ('bool', False)
        if t == 'NULL': self.eat('NULL'); return ('null', None)
        if t == 'THIS': self.eat('THIS'); return ('this',)
        if t == TT_IDENT:
            name = self.cur.value; self.eat(TT_IDENT)
            return ('var', name)
        if t == 'MINUS':
            self.eat('MINUS')
            return ('un', 'NEG', self.parse_expr(7))
        if t == 'BANG':
            self.eat('BANG')
            return ('un', 'NOT', self.parse_expr(7))
        if t == 'LP':
            self.eat('LP')
            e = self.parse_expr()
            self.eat('RP')
            return e
        if t == 'LBR':
            self.eat('LBR')
            items = []
            if self.cur.type != 'RBR':
                while True:
                    items.append(self.parse_expr())
                    if not self.match('COMMA'): break
            self.eat('RBR')
            return ('arr', items)
        if t == 'LB':
            self.eat('LB')
            pairs = []
            if self.cur.type != 'RB':
                while True:
                    if self.cur.type == TT_STRING:
                        k = self.cur.value; self.eat(TT_STRING)
                    elif self.cur.type == TT_IDENT:
                        k = self.cur.value; self.eat(TT_IDENT)
                    else:
                        raise SyntaxError("Klucz mapy musi być identyfikatorem lub napisem")
                    self.eat('COLON')
                    v = self.parse_expr()
                    pairs.append((k, v))
                    if not self.match('COMMA'): break
            self.eat('RB')
            return ('map', pairs)
        if t == 'FUNC':
            return self.func_literal()
        if t == 'NEW':
            self.eat('NEW')
            cls = self.parse_expr(7)
            self.eat('LP')
            args = []
            if self.cur.type != 'RP':
                while True:
                    args.append(self.parse_expr())
                    if not self.match('COMMA'): break
            self.eat('RP')
            return ('new', cls, args)
        raise SyntaxError(f"Nieoczekiwany token w wyrażeniu: {t}")

    def parse_args(self) -> List[Any]:
        self.eat('LP')
        args = []
        if self.cur.type != 'RP':
            while True:
                args.append(self.parse_expr())
                if not self.match('COMMA'): break
        self.eat('RP')
        return args

    def func_literal(self):
        self.eat('FUNC')
        name = None
        if self.cur.type == TT_IDENT:
            name = self.cur.value
            self.eat(TT_IDENT)
        self.eat('LP')
        params = []
        if self.cur.type != 'RP':
            while True:
                params.append(self.eat(TT_IDENT).value)
                if not self.match('COMMA'): break
        self.eat('RP')
        body = self.block()
        return ('func', name, params, body)

    # ------------- Instrukcje -------------
    def statement(self):
        t = self.cur.type
        if t == 'LET':
            self.eat('LET')
            name = self.eat(TT_IDENT).value
            self.eat('ASSIGN')
            expr = self.parse_expr()
            self.match('SC')
            return ('let', name, expr)
        if t == 'IF':
            self.eat('IF')
            self.eat('LP'); cond = self.parse_expr(); self.eat('RP')
            th = self.block()
            el = None
            if self.match('ELSE'):
                el = self.block()
            return ('if', cond, th, el)
        if t == 'WHILE':
            self.eat('WHILE')
            self.eat('LP'); cond = self.parse_expr(); self.eat('RP')
            body = self.block()
            return ('while', cond, body)
        if t == 'RETURN':
            self.eat('RETURN')
            expr = None
            if self.cur.type != 'SC' and self.cur.type != 'RB':
                expr = self.parse_expr()
            self.match('SC')
            return ('return', expr)
        if t == 'BREAK':
            self.eat('BREAK'); self.match('SC'); return ('break',)
        if t == 'CONTINUE':
            self.eat('CONTINUE'); self.match('SC'); return ('continue',)
        if t == 'LB':
            return self.block()
        if t == 'FOR':
            self.eat('FOR')
            self.eat('LP')
            self.eat('LET')
            var = self.eat(TT_IDENT).value
            self.eat('IN')
            iterable = self.parse_expr()
            self.eat('RP')
            body = self.block()
            return ('forin', var, iterable, body)
        if t == 'IMPORT':
            self.eat('IMPORT')
            path = self.eat(TT_STRING).value
            self.match('SC')
            return ('import', path)
        if t == 'CLASS':
            self.eat('CLASS')
            name = self.eat(TT_IDENT).value
            body = self.block()
            methods = []
            for st in body[1]:
                if st[0] == 'expr' and st[1][0] == 'func':
                    fn = st[1]
                elif st[0] == 'func':
                    fn = st
                else:
                    raise SyntaxError("W klasie dozwolone są tylko definicje 'func'")
                _, mname, params, mbody = fn
                if mname is None:
                    raise SyntaxError("Metoda w klasie musi mieć nazwę")
                methods.append((mname, params, mbody))
            return ('class', name, methods)
        if t == 'THIS':
            self.eat('THIS'); self.eat('DOT')
            prop = self.eat(TT_IDENT).value
            if self.cur.type in ('ASSIGN','PLUS_EQ','MINUS_EQ','STAR_EQ','SLASH_EQ','MOD_EQ'):
                op = self.cur.type; self.eat(op)
                expr = self.parse_expr()
                self.match('SC')
                return ('setprop', ('this',), prop, op, expr)
            else:
                raise SyntaxError("Oczekiwano przypisania do this.prop")
        if t == TT_IDENT:
            name = self.eat(TT_IDENT).value
            if self.cur.type in ('ASSIGN','PLUS_EQ','MINUS_EQ','STAR_EQ','SLASH_EQ','MOD_EQ'):
                op = self.cur.type; self.eat(op)
                expr = self.parse_expr()
                self.match('SC')
                return ('assign_op', name, op, expr)
            # backtrack
            self.lx.i -= len(name)
            self.cur = Token(TT_IDENT, name, self.lx.i)
        expr = self.parse_expr()
        self.match('SC')
        return ('expr', expr)

    def block(self):
        self.eat('LB')
        stmts = []
        while self.cur.type != 'RB':
            stmts.append(self.statement())
        self.eat('RB')
        return ('block', stmts)

    def program(self):
        stmts = []
        while self.cur.type != TT_EOF:
            stmts.append(self.statement())
        return ('block', stmts)

############################################################
# RUNTIME
############################################################

class BreakEx(Exception): pass
class ContinueEx(Exception): pass
class ReturnEx(Exception):
    def __init__(self, value): self.value = value

class Env:
    def __init__(self, parent: Optional['Env']=None):
        self.parent = parent
        self.values: Dict[str, Any] = {}
    def get(self, name: str):
        if name in self.values: return self.values[name]
        if self.parent: return self.parent.get(name)
        raise NameError(f"Niezdefiniowana zmienna: {name}")
    def set(self, name: str, val: Any):
        if name in self.values:
            self.values[name] = val; return
        if self.parent and name in self.parent:
            self.parent.set(name, val); return
        self.values[name] = val
    def __contains__(self, key):
        if key in self.values: return True
        return key in self.parent if self.parent else False

from dataclasses import dataclass

@dataclass
class Function:
    name: Optional[str]
    params: List[str]
    body: Any
    env: Env
    def __call__(self, args: List[Any]):
        if len(args) != len(self.params):
            raise TypeError(f"Funkcja oczekuje {len(self.params)} argumentów, podano {len(args)}")
        local = Env(self.env)
        for p, a in zip(self.params, args):
            local.values[p] = a
        try:
            Interpreter.eval_block(self.body, local)
        except ReturnEx as r:
            return r.value
        return None
    def __repr__(self):
        n = self.name or "<anon>"
        return f"<func {n}/{len(self.params)}>"

class BoundMethod:
    def __init__(self, func: 'Function', this_obj: Any):
        self.func = func
        self.this = this_obj
    def __call__(self, args: List[Any]):
        local = Env(self.func.env)
        local.values.update({'this': self.this})
        if len(args) != len(self.func.params):
            raise TypeError(f"Funkcja oczekuje {len(self.func.params)} argumentów, podano {len(self.func.params)}")
        for p, a in zip(self.func.params, args):
            local.values[p] = a
        try:
            Interpreter.eval_block(self.func.body, local)
        except ReturnEx as r:
            return r.value
        return None

class Interpreter:
    module_cache: Dict[str, Any] = {}

    @staticmethod
    def truthy(v):
        return bool(v)

    @staticmethod
    def _interp_string(s: str, env: 'Env') -> str:
        out = []
        i = 0
        while i < len(s):
            if s[i] == '$' and i+1 < len(s) and s[i+1] == '{':
                j = i+2
                depth = 1
                buf = []
                while j < len(s) and depth > 0:
                    if s[j] == '{': depth += 1
                    elif s[j] == '}':
                        depth -= 1
                        if depth == 0:
                            break
                    if depth > 0:
                        buf.append(s[j])
                    j += 1
                expr_src = ''.join(buf)
                val = Interpreter.eval(Parser(Lexer(expr_src)).parse_expr(), env)
                out.append(str(val))
                i = j + 1
            else:
                out.append(s[i]); i += 1
        return ''.join(out)

    @staticmethod
    def eval(node, env: Env):
        t = node[0]
        if t == 'block':
            return Interpreter.eval_block(node, Env(env))
        if t == 'let':
            _, name, expr = node
            env.values[name] = Interpreter.eval(expr, env)
            return None
        if t == 'assign':
            _, name, expr = node
            val = Interpreter.eval(expr, env)
            if name in env:
                e = env
                while e and name not in e.values:
                    e = e.parent
                e.values[name] = val
            else:
                env.values[name] = val
            return val
        if t == 'assign_op':
            _, name, op, expr = node
            cur = env.get(name) if name in env else None
            val = Interpreter.eval(expr, env)
            if op == 'ASSIGN': newv = val
            elif op == 'PLUS_EQ': newv = Interpreter._arith('+', cur, val)
            elif op == 'MINUS_EQ': newv = Interpreter._arith('-', cur, val)
            elif op == 'STAR_EQ': newv = Interpreter._arith('*', cur, val)
            elif op == 'SLASH_EQ': newv = Interpreter._arith('/', cur, val)
            elif op == 'MOD_EQ': newv = Interpreter._arith('%', cur, val)
            else: raise RuntimeError('Nieznany operator przypisania')
            e = env
            if name in e:
                while e and name not in e.values:
                    e = e.parent
                e.values[name] = newv
            else:
                env.values[name] = newv
            return newv
        if t == 'setprop':
            _, this_node, prop, op, expr = node
            obj = Interpreter.eval(this_node, env)
            cur = obj.get('__fields__', {}).get(prop, None)
            val = Interpreter.eval(expr, env)
            if op == 'ASSIGN': newv = val
            elif op == 'PLUS_EQ': newv = Interpreter._arith('+', cur, val)
            elif op == 'MINUS_EQ': newv = Interpreter._arith('-', cur, val)
            elif op == 'STAR_EQ': newv = Interpreter._arith('*', cur, val)
            elif op == 'SLASH_EQ': newv = Interpreter._arith('/', cur, val)
            elif op == 'MOD_EQ': newv = Interpreter._arith('%', cur, val)
            else: raise RuntimeError('Nieznany operator przypisania')
            obj.setdefault('__fields__', {})[prop] = newv
            return newv
        if t == 'expr':
            return Interpreter.eval(node[1], env)
        if t == 'if':
            _, cond, th, el = node
            if Interpreter.truthy(Interpreter.eval(cond, env)):
                return Interpreter.eval_block(th, Env(env))
            if el:
                return Interpreter.eval_block(el, Env(env))
            return None
        if t == 'while':
            _, cond, body = node
            while Interpreter.truthy(Interpreter.eval(cond, env)):
                try:
                    Interpreter.eval_block(body, Env(env))
                except BreakEx:
                    break
                except ContinueEx:
                    continue
            return None
        if t == 'forin':
            _, var, iterable, body = node
            it = Interpreter.eval(iterable, env)
            for v in it:
                local = Env(env)
                local.values[var] = v
                try:
                    Interpreter.eval_block(body, local)
                except BreakEx:
                    break
                except ContinueEx:
                    continue
            return None
        if t == 'import':
            _, path = node
            if path in Interpreter.module_cache:
                return None
            with open(path, 'r', encoding='utf-8') as f:
                src = f.read()
            lx = Lexer(src); ps = Parser(lx)
            prog = ps.program()
            Interpreter.module_cache[path] = True
            Interpreter.eval_block(prog, env)
            return None
        if t == 'class':
            _, name, methods = node
            meth_map = {}
            ctor = None
            for (mname, params, body) in methods:
                fn = Function(mname, params, body, Env(env))
                if mname == 'init':
                    ctor = fn
                else:
                    meth_map[mname] = fn
            cls = {'__name__': name, '__init__': ctor, '__methods__': meth_map}
            env.values[name] = cls
            return cls
        if t == 'new':
            _, cls_node, args = node
            cls = Interpreter.eval(cls_node, env)
            inst = {'__class__': cls, '__fields__': {}}
            ctor = cls.get('__init__') if isinstance(cls, dict) else None
            if ctor:
                bm = BoundMethod(ctor, inst)
                bm(args=[Interpreter.eval(a, env) for a in args])
            return inst
        if t == 'return':
            _, expr = node
            val = Interpreter.eval(expr, env) if expr is not None else None
            raise ReturnEx(val)
        if t == 'break':
            raise BreakEx()
        if t == 'continue':
            raise ContinueEx()
        if t == 'num': return node[1]
        if t == 'str':
            return Interpreter._interp_string(node[1], env)
        if t == 'bool': return node[1]
        if t == 'null': return None
        if t == 'this':
            return env.get('this')
        if t == 'var':
            return env.get(node[1])
        if t == 'un':
            _, op, e = node
            v = Interpreter.eval(e, env)
            if op == 'NEG': return -float(v) if isinstance(v, float) else -int(v)
            if op == 'NOT': return not Interpreter.truthy(v)
        if t == 'bin':
            _, op, l, r = node
            if op in ('AND', 'OR'):
                lv = Interpreter.eval(l, env)
                if op == 'AND':
                    return Interpreter.eval(r, env) if Interpreter.truthy(lv) else False
                else:
                    return lv if Interpreter.truthy(lv) else Interpreter.eval(r, env)
            lv = Interpreter.eval(l, env)
            rv = Interpreter.eval(r, env)
            return Interpreter._arith({'PLUS':'+','MINUS':'-','STAR':'*','SLASH':'/','MOD':'%','EQ':'==','NE':'!=','LT':'<','LE':'<=','GT':'>','GE':'>='}[op], lv, rv)
        if t == 'arr':
            return [Interpreter.eval(e, env) for e in node[1]]
        if t == 'map':
            return {k: Interpreter.eval(v, env) for (k,v) in node[1]}
        if t == 'index':
            _, coll, idx = node
            c = Interpreter.eval(coll, env)
            i = Interpreter.eval(idx, env)
            return c[i]
        if t == 'getprop':
            _, obj_node, name = node
            obj = Interpreter.eval(obj_node, env)
            if isinstance(obj, dict) and '__class__' in obj:
                if name in obj['__fields__']:
                    return obj['__fields__'][name]
                meth = obj['__class__']['__methods__'].get(name)
                if meth:
                    return BoundMethod(meth, obj)
                raise AttributeError(f"Brak pola/metody: {name}")
            if isinstance(obj, dict):
                return obj.get(name)
            raise TypeError('Kropka: obiekt nieobsługiwalny')
        if t == 'call':
            _, callee, args = node
            f = Interpreter.eval(callee, env)
            a = [Interpreter.eval(x, env) for x in args]
            if isinstance(f, Function):
                return f(a)
            if isinstance(f, BoundMethod):
                return f(a)
            if callable(f):
                return f(*a)
            raise TypeError("Obiekt nie jest wywoływalny")
        if t == 'func':
            _, name, params, body = node
            fn = Function(name, params, body, Env(env))
            if name:
                env.values[name] = fn
            return fn
        raise RuntimeError(f"Nieobsługiwany node: {t}")

    @staticmethod
    def _arith(op, lv, rv=None):
        if op in ('==','!=','<','<=','>','>='):
            if op == '==': return lv == rv
            if op == '!=': return lv != rv
            if op == '<': return lv < rv
            if op == '<=': return lv <= rv
            if op == '>': return lv > rv
            if op == '>=': return lv >= rv
        if op == '+':
            if isinstance(lv, str) or isinstance(rv, str):
                return str(lv) + str(rv)
            if isinstance(lv, int) and isinstance(rv, int):
                return lv + rv
            return float(lv) + float(rv)
        if op == '-':
            if isinstance(lv, int) and isinstance(rv, int): return lv - rv
            return float(lv) - float(rv)
        if op == '*':
            if isinstance(lv, int) and isinstance(rv, int): return lv * rv
            return float(lv) * float(rv)
        if op == '/':
            return float(lv) / float(rv)
        if op == '%':
            if isinstance(lv, int) and isinstance(rv, int): return lv % rv
            return float(lv) % float(rv)
        raise RuntimeError('Nieznana operacja')

    @staticmethod
    def eval_block(block_node, env: Env):
        assert block_node[0] == 'block'
        for st in block_node[1]:
            Interpreter.eval(st, env)
        return None

############################################################
# BUILTINS
############################################################

def make_builtins() -> Dict[str, Any]:
    def _print(*args): print(*args)
    def _len(x): return len(x)
    def _range(a, b=None, step=1):
        if b is None: start, end = 0, int(a)
        else: start, end = int(a), int(b)
        return list(range(start, end, int(step)))
    def _type(x):
        if x is None: return 'null'
        if isinstance(x, bool): return 'bool'
        if isinstance(x, int): return 'int'
        if isinstance(x, float): return 'number'
        if isinstance(x, str): return 'string'
        if isinstance(x, list): return 'array'
        if isinstance(x, dict):
            if '__class__' in x: return 'instance'
            return 'map'
        if isinstance(x, Function): return 'function'
        return type(x).__name__
    def _keys(m): return list(m.keys())
    def _values(m): return list(m.values())
    def _push(arr, v): arr.append(v); return arr
    def _pop(arr): return arr.pop()
    def _str(x): return str(x)
    def _num(x): return float(x)
    def _readline(prompt=""): return input(prompt)
    def _join(arr, sep=""): return str(sep).join(map(str, arr))
    def _split(s, sep=None): return str(s).split(sep)
    def _map(fn, arr):
        out = []
        for v in arr:
            out.append(fn(v) if callable(fn) else v)
        return out
    def _filter(fn, arr):
        out = []
        for v in arr:
            keep = fn(v) if callable(fn) else v
            if keep: out.append(v)
        return out
    def _get(obj, key):
        if isinstance(obj, dict) and '__class__' in obj:
            return obj['__fields__'].get(key)
        if isinstance(obj, dict): return obj.get(key)
        raise TypeError('get: nieobsługiwalny obiekt')
    def _set(obj, key, value):
        if isinstance(obj, dict) and '__class__' in obj:
            obj['__fields__'][key] = value; return value
        if isinstance(obj, dict):
            obj[key] = value; return value
        raise TypeError('set: nieobsługiwalny obiekt')
    return {
        'print': _print, 'len': _len, 'range': _range, 'type': _type,
        'keys': _keys, 'values': _values, 'push': _push, 'pop': _pop,
        'str': _str, 'num': _num, 'readline': _readline, 'join': _join,
        'split': _split, 'map': _map, 'filter': _filter, 'get': _get, 'set': _set,
    }

############################################################
# REPL / Plik
############################################################

def run(src: str, *, filename: str='<stdin>'):
    lx = Lexer(src); ps = Parser(lx); prog = ps.program()
    global_env = Env(); global_env.values.update(make_builtins())
    Interpreter.eval_block(prog, global_env)
    return global_env

SAMPLE = r'''
// Nowe możliwości w Borsuku
print("Interpolacja: ${1+2} i ${""+str(3)}")

class Osoba {
  func init(imie, wiek) { this.imie = imie; this.wiek = wiek; }
  func hello() { print("Cześć, jestem ${this.imie} (${this.wiek})"); }
}

let p = new Osoba("Ala", 20);
p.hello();

let x = 5; x += 2; x *= 3; print("x=", x); // 21

let m = {a:1}; set(m, "b", 2); print("m keys:", keys(m));
'''

def repl():
    env = Env(); env.values.update(make_builtins())
    buf = ""
    print("Borsuk REPL. Zakończ Ctrl-D/Ctrl-C. Puste wiersze wykonują kod.")
    while True:
        try:
            line = input(".. ")
        except EOFError:
            print() ; break
        except KeyboardInterrupt:
            print("\n^C") ; break
        if line.strip() == "":
            if not buf.strip():
                continue
            try:
                lx = Lexer(buf); ps = Parser(lx); prog = ps.program()
                Interpreter.eval_block(prog, env)
            except Exception as e:
                print("[Błąd]", e)
            buf = ""
        else:
            buf += line + "\n"

if __name__ == '__main__':
    if len(sys.argv) == 1:
        repl()
    else:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            src = f.read()
        try:
            run(src, filename=sys.argv[1])
        except Exception as e:
            print("[Błąd]", e)
            sys.exit(1)

# From DIRTY/DIRE utils:
# https://github.com/CMUSTRUDEL/DIRTY/blob/main/dire/utils/lexer.py


from enum import Enum, auto
from pygments import lex
from pygments.token import Token
from pygments.token import is_token_subtype
from pygments.lexers.c_cpp import CLexer, inherit

Token.Placeholder = Token.Token.Placeholder


class Names(Enum):
    RAW = auto()
    SOURCE = auto()
    TARGET = auto()


class TokenError(Exception):
    def __init__(self, message):
        self.message = message


class Lexer:
    def __init__(self, raw_code):
        self.raw_code = raw_code
        self.tokens = list(lex(self.raw_code, HexRaysLexer()))

    def get_tokens(self, var_names=Names.RAW):
        """Generate tokens from a raw_code string, skipping comments.

        Keyword arguments:
        var_names -- Which variable names to output (default RAW).
        """
        previous_string = None
        for (token_type, token) in self.tokens:
            # Pygments breaks up strings into individual tokens representing
            # things like opening quotes and escaped characters. We want to
            # collapse all of these into a single string literal token.
            if previous_string and not is_token_subtype(token_type, Token.String):
                yield (Token.String, previous_string)
                previous_string = None
            if is_token_subtype(token_type, Token.String):
                if previous_string:
                    previous_string += token
                else:
                    previous_string = token
            elif is_token_subtype(token_type, Token.Number):
                yield (token_type, token)
            # Skip comments
            elif is_token_subtype(token_type, Token.Comment):
                continue
            # Skip the :: token added by HexRays
            elif is_token_subtype(token_type, Token.Operator) and token == '::':
                continue
            # Replace the text of placeholder tokens
            elif is_token_subtype(token_type, Token.Placeholder):
                yield {
                    Names.RAW : (token_type, token),
                    Names.SOURCE : (token_type, token.split('@@')[2]),
                    Names.TARGET : (token_type, token.split('@@')[3]),
                }[var_names]
            elif not is_token_subtype(token_type, Token.Text):
                yield (token_type, token.strip())
            # Skip whitespace
            elif is_token_subtype(token_type, Token.Text):
                continue
            else:
                raise TokenError(f"No token ({token_type}, {token})")


class HexRaysLexer(CLexer):
    # Additional tokens
    tokens = {
        'statements' : [
            (r'->', Token.Operator),
            (r'\+\+', Token.Operator),
            (r'--', Token.Operator),
            (r'==', Token.Operator),
            (r'!=', Token.Operator),
            (r'>=', Token.Operator),
            (r'<=', Token.Operator),
            (r'&&', Token.Operator),
            (r'\|\|', Token.Operator),
            (r'\+=', Token.Operator),
            (r'-=', Token.Operator),
            (r'\*=', Token.Operator),
            (r'/=', Token.Operator),
            (r'%=', Token.Operator),
            (r'&=', Token.Operator),
            (r'\^=', Token.Operator),
            (r'\|=', Token.Operator),
            (r'<<=', Token.Operator),
            (r'>>=', Token.Operator),
            (r'<<', Token.Operator),
            (r'>>', Token.Operator),
            (r'\.\.\.', Token.Operator),
            (r'##', Token.Operator),
            (r'::', Token.Operator),
            (r'@@\w+@@\w+@@', Token.Placeholder.Var),
            inherit
        ]
    }

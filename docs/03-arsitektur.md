# SimpleNeural-DSL: Arsitektur Sistem dan Data Model

## 7. Entity Relationship Diagram (ERD)

### 7.1 Conceptual ERD - Domain Model

```mermaid
erDiagram
    PROGRAM ||--|| DATASET_DECL : contains
    PROGRAM ||--|{ MODEL_DECL : contains

    DATASET_DECL {
        string file_path PK
        string target_column
        string delimiter
        boolean has_header
    }

    MODEL_DECL ||--|{ LAYER_DECL : contains
    MODEL_DECL ||--|| OPTIMIZER_DECL : has
    MODEL_DECL ||--|| TRAIN_CONFIG : has

    MODEL_DECL {
        string model_name PK
        string loss_function
        string[] metrics
    }

    LAYER_DECL {
        int layer_index PK
        string layer_type
        int units
        string activation
        float dropout_rate
    }

    OPTIMIZER_DECL {
        string optimizer_type PK
        float learning_rate
        float momentum
        float beta1
        float beta2
    }

    TRAIN_CONFIG {
        int epochs
        int batch_size
        float validation_split
        boolean shuffle
        string callbacks
    }

    LAYER_DECL ||--o| LAYER_PARAMS : has_optional

    LAYER_PARAMS {
        string param_name PK
        string param_value
        string param_type
    }
```

### 7.2 Internal Data Structures ERD

```mermaid
erDiagram
    TOKEN ||--|| POSITION : has
    TOKEN {
        string token_type PK
        string value
        int line
        int column
    }

    POSITION {
        int line PK
        int column PK
        int offset
    }

    AST_NODE ||--o{ AST_NODE : has_children
    AST_NODE {
        string node_type PK
        string value
        int source_line
        int source_column
    }

    SYMBOL_TABLE ||--|{ SYMBOL_ENTRY : contains

    SYMBOL_TABLE {
        string scope_name PK
        string parent_scope FK
    }

    SYMBOL_ENTRY {
        string name PK
        string symbol_type
        string data_type
        int line_defined
        boolean is_used
    }

    ERROR_INFO {
        string error_type PK
        string message
        int line
        int column
        string suggestion
        string source_snippet
    }
```

### 7.3 Compiler Pipeline Data Model

```mermaid
erDiagram
    SOURCE_FILE ||--|| TOKEN_STREAM : produces
    TOKEN_STREAM ||--|| PARSE_TREE : produces
    PARSE_TREE ||--|| AST : produces
    AST ||--|| SYMBOL_TABLE : produces
    AST ||--|| PYTHON_CODE : produces

    SOURCE_FILE {
        string file_path PK
        string content
        string encoding
        int line_count
    }

    TOKEN_STREAM {
        int stream_id PK
        Token[] tokens
        int token_count
    }

    PARSE_TREE {
        int tree_id PK
        Node root_node
        int node_count
    }

    AST {
        int ast_id PK
        ASTNode root
        boolean is_valid
    }

    SYMBOL_TABLE {
        string scope_id PK
        Entry[] entries
        string[] scopes
    }

    PYTHON_CODE {
        string output_path PK
        string code_content
        int line_count
        string[] imports
    }
```

---

## 8. Arsitektur Sistem

### 8.1 System Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Interface]
        API[Python API]
    end

    subgraph "Compiler Core"
        LEX[Lexer<br/>Lexical Analyzer]
        PAR[Parser<br/>Syntax Analyzer]
        SEM[Semantic Analyzer<br/>Type Checker]
        GEN[Code Generator]
    end

    subgraph "Supporting Modules"
        ERR[Error Handler]
        SYM[Symbol Table Manager]
        TPL[Code Template Engine]
    end

    subgraph "Output Layer"
        PY[Python Code Output]
        LOG[Compilation Log]
    end

    subgraph "External Dependencies"
        TF[TensorFlow/Keras]
        PD[Pandas]
    end

    CLI --> LEX
    API --> LEX

    LEX --> PAR
    PAR --> SEM
    SEM --> GEN

    LEX -.-> ERR
    PAR -.-> ERR
    SEM -.-> ERR

    PAR --> SYM
    SEM --> SYM

    GEN --> TPL
    GEN --> PY

    ERR --> LOG

    PY -.-> TF
    PY -.-> PD

    style LEX fill:#e7f3ff
    style PAR fill:#e7f3ff
    style SEM fill:#e7f3ff
    style GEN fill:#e7f3ff
```

### 8.2 Component Diagram

```mermaid
graph LR
    subgraph "simpleneural Package"
        subgraph "lexer/"
            L1[lexer.py]
            L2[tokens.py]
            L3[patterns.py]
        end

        subgraph "parser/"
            P1[parser.py]
            P2[grammar.py]
            P3[ast_nodes.py]
        end

        subgraph "semantic/"
            S1[analyzer.py]
            S2[type_checker.py]
            S3[symbol_table.py]
        end

        subgraph "codegen/"
            G1[generator.py]
            G2[templates.py]
            G3[formatter.py]
        end

        subgraph "common/"
            C1[errors.py]
            C2[utils.py]
            C3[config.py]
        end

        MAIN[main.py]
    end

    L1 --> P1
    P1 --> S1
    S1 --> G1

    L2 --> L1
    L3 --> L1
    P2 --> P1
    P3 --> P1
    S2 --> S1
    S3 --> S1
    G2 --> G1
    G3 --> G1

    C1 --> L1
    C1 --> P1
    C1 --> S1

    MAIN --> L1
    MAIN --> P1
    MAIN --> S1
    MAIN --> G1
```

### 8.3 Sequence Diagram - Kompilasi

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Lexer
    participant Parser
    participant Semantic
    participant CodeGen
    participant FileSystem

    User->>CLI: simpleneural compile model.sndsl
    CLI->>FileSystem: Read source file
    FileSystem-->>CLI: Source content

    CLI->>Lexer: tokenize(source)

    loop For each character
        Lexer->>Lexer: Match pattern
        Lexer->>Lexer: Create token
    end

    Lexer-->>CLI: Token stream

    CLI->>Parser: parse(tokens)

    loop For each grammar rule
        Parser->>Parser: Match rule
        Parser->>Parser: Build tree node
    end

    Parser-->>CLI: Parse tree / AST

    CLI->>Semantic: analyze(ast)
    Semantic->>Semantic: Build symbol table
    Semantic->>Semantic: Type check
    Semantic->>Semantic: Validate references
    Semantic-->>CLI: Validated AST

    CLI->>CodeGen: generate(ast)
    CodeGen->>CodeGen: Generate imports
    CodeGen->>CodeGen: Generate data loading
    CodeGen->>CodeGen: Generate model
    CodeGen->>CodeGen: Generate training
    CodeGen-->>CLI: Python code string

    CLI->>FileSystem: Write output.py
    FileSystem-->>CLI: Success

    CLI-->>User: Compilation successful!
```

### 8.4 Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input"
        A[Source .sndsl]
    end

    subgraph "Processing"
        B[Character Stream]
        C[Token Stream]
        D[Parse Tree]
        E[AST]
        F[Annotated AST]
        G[Code Fragments]
    end

    subgraph "Output"
        H[Python .py]
    end

    A -->|"Read File"| B
    B -->|"Lexer<br/>(Regex/DFA)"| C
    C -->|"Parser<br/>(CFG)"| D
    D -->|"Tree Transform"| E
    E -->|"Type Check<br/>Validation"| F
    F -->|"Template<br/>Substitution"| G
    G -->|"Format &<br/>Write"| H
```

---

## 9. Deployment Architecture

### 9.1 Deployment Diagram

```mermaid
graph TB
    subgraph "User Machine"
        subgraph "Python Environment"
            SN[SimpleNeural Package]
            PY[Python 3.8+]
            PIP[pip packages]
        end

        subgraph "File System"
            SRC[Source Files .sndsl]
            OUT[Output Files .py]
            DATA[Dataset Files .csv]
        end

        subgraph "ML Runtime"
            TF[TensorFlow 2.x]
            NP[NumPy]
            PD[Pandas]
        end
    end

    SN --> PY
    PIP --> TF
    PIP --> NP
    PIP --> PD

    SRC --> SN
    SN --> OUT
    OUT --> TF
    DATA --> OUT

    style SN fill:#4ecdc4
    style TF fill:#ff6b35
```

### 9.2 Package Structure

```
simpleneural/
├── __init__.py
├── __main__.py              # Entry point untuk `python -m simpleneural`
├── main.py                  # CLI interface
│
├── lexer/
│   ├── __init__.py
│   ├── lexer.py            # Main lexer class
│   ├── tokens.py           # Token type definitions
│   └── patterns.py         # Regex patterns
│
├── parser/
│   ├── __init__.py
│   ├── parser.py           # Main parser class
│   ├── grammar.py          # CFG rules definition
│   └── ast_nodes.py        # AST node classes
│
├── semantic/
│   ├── __init__.py
│   ├── analyzer.py         # Main semantic analyzer
│   ├── type_checker.py     # Type validation
│   └── symbol_table.py     # Symbol table management
│
├── codegen/
│   ├── __init__.py
│   ├── generator.py        # Main code generator
│   ├── templates.py        # Code templates
│   └── formatter.py        # Python code formatter
│
├── common/
│   ├── __init__.py
│   ├── errors.py           # Exception classes
│   ├── utils.py            # Utility functions
│   └── config.py           # Configuration
│
└── templates/
    ├── tensorflow.py.template
    ├── keras.py.template
    └── sklearn.py.template
```

---

## 10. Class Diagram

### 10.1 Core Classes

```mermaid
classDiagram
    class Compiler {
        -lexer: Lexer
        -parser: Parser
        -semantic: SemanticAnalyzer
        -codegen: CodeGenerator
        +compile(source: str): str
        +compile_file(path: str): None
        +validate(source: str): List~Error~
    }

    class Lexer {
        -patterns: Dict~str, Pattern~
        -position: Position
        -source: str
        +tokenize(source: str): List~Token~
        -match_token(): Token
        -skip_whitespace(): None
        -skip_comment(): None
    }

    class Token {
        +type: TokenType
        +value: str
        +line: int
        +column: int
        +__repr__(): str
    }

    class Parser {
        -tokens: List~Token~
        -current: int
        -grammar: Grammar
        +parse(tokens: List~Token~): AST
        -parse_program(): ProgramNode
        -parse_dataset(): DatasetNode
        -parse_model(): ModelNode
        -parse_layer(): LayerNode
        -expect(type: TokenType): Token
        -match(type: TokenType): bool
    }

    class ASTNode {
        <<abstract>>
        +node_type: str
        +line: int
        +column: int
        +accept(visitor: Visitor): Any
    }

    class SemanticAnalyzer {
        -symbol_table: SymbolTable
        -errors: List~Error~
        +analyze(ast: AST): AST
        -check_types(): None
        -check_references(): None
        -validate_model(): None
    }

    class CodeGenerator {
        -templates: TemplateEngine
        -formatter: Formatter
        +generate(ast: AST): str
        -gen_imports(): str
        -gen_dataset(): str
        -gen_model(): str
        -gen_training(): str
    }

    Compiler --> Lexer
    Compiler --> Parser
    Compiler --> SemanticAnalyzer
    Compiler --> CodeGenerator
    Lexer --> Token
    Parser --> ASTNode
    SemanticAnalyzer --> ASTNode
    CodeGenerator --> ASTNode
```

### 10.2 AST Node Hierarchy

```mermaid
classDiagram
    class ASTNode {
        <<abstract>>
        +line: int
        +column: int
        +accept(visitor): Any
    }

    class ProgramNode {
        +dataset: DatasetNode
        +models: List~ModelNode~
    }

    class DatasetNode {
        +file_path: str
        +target_column: str
        +options: Dict
    }

    class ModelNode {
        +name: str
        +layers: List~LayerNode~
        +optimizer: OptimizerNode
        +train_config: TrainNode
    }

    class LayerNode {
        +layer_type: str
        +params: Dict
    }

    class OptimizerNode {
        +optimizer_type: str
        +learning_rate: float
        +params: Dict
    }

    class TrainNode {
        +epochs: int
        +batch_size: int
        +validation_split: float
    }

    ASTNode <|-- ProgramNode
    ASTNode <|-- DatasetNode
    ASTNode <|-- ModelNode
    ASTNode <|-- LayerNode
    ASTNode <|-- OptimizerNode
    ASTNode <|-- TrainNode

    ProgramNode *-- DatasetNode
    ProgramNode *-- ModelNode
    ModelNode *-- LayerNode
    ModelNode *-- OptimizerNode
    ModelNode *-- TrainNode
```

### 10.3 Error Handling Classes

```mermaid
classDiagram
    class CompilerError {
        <<abstract>>
        +message: str
        +line: int
        +column: int
        +source_snippet: str
        +format(): str
    }

    class LexicalError {
        +invalid_char: str
        +expected_pattern: str
    }

    class SyntaxError {
        +expected_token: str
        +found_token: str
        +grammar_rule: str
    }

    class SemanticError {
        +error_type: str
        +symbol_name: str
        +suggestion: str
    }

    class TypeError {
        +expected_type: str
        +actual_type: str
        +context: str
    }

    class ReferenceError {
        +undefined_symbol: str
        +available_symbols: List~str~
    }

    CompilerError <|-- LexicalError
    CompilerError <|-- SyntaxError
    CompilerError <|-- SemanticError
    SemanticError <|-- TypeError
    SemanticError <|-- ReferenceError
```

---

_Dokumen ini adalah bagian ketiga dari rancangan SimpleNeural-DSL. Lanjut ke dokumen berikutnya untuk Grammar & Token Specification._

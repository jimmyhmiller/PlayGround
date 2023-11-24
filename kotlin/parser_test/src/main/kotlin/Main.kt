import kotlinx.serialization.*
import kotlinx.serialization.json.*



@Serializable
@SerialName("Program")
data class Program(
    val type: String,
    val body: List<Expression>,
    val sourceType: String,
    val range: List<Int>
)

@Serializable
@SerialName("ExpressionStatement")
data class ExpressionStatement(
    val type: String,
    val expression: Expression,
    val range: List<Int>
) : Expression()

@Serializable
sealed class Expression

@Serializable
@SerialName("BinaryExpression")
data class BinaryExpression(
    val operator: String,
    val left: Expression,
    val right: Expression,
    val range: List<Int>
) : Expression()

@Serializable
@SerialName("Literal")
data class Literal(
    val value: Int,
    val raw: String,
    val range: List<Int>
) : Expression()

@Serializable
@SerialName("Identifier")
data class Identifier(
    val name: String,
    val range: List<Int>
) : Expression()

@Serializable
@SerialName("CallExpression")
data class CallExpression(
    val callee: Expression,
    val arguments: List<Expression>,
    val range: List<Int>
) : Expression()

@Serializable
@SerialName("MemberExpression")
data class MemberExpression(
    val `object`: Expression,
    val property: Expression,
    val computed: Boolean,
    val range: List<Int>
) : Expression()

@Serializable
@SerialName("FunctionExpression")
data class FunctionExpression(
    val id: Identifier?,
    val params: List<Identifier>,
    val body: BlockStatement,
    val range: List<Int>
) : Expression()

@Serializable
@SerialName("BlockStatement")
data class BlockStatement(
    val type: String,
    val body: List<Statement>,
    val range: List<Int>
)

@Serializable
sealed class Statement : Expression()

@Serializable
@SerialName("VariableDeclaration")
data class VariableDeclaration(
    val declarations: List<VariableDeclarator>,
    val kind: String,
    val range: List<Int>
) : Statement()

@Serializable
@SerialName("VariableDeclarator")
data class VariableDeclarator(
    val id: Identifier,
    val init: Expression?,
    val range: List<Int>
)

@Serializable
@SerialName("IfStatement")
data class IfStatement(
    val test: Expression,
    val consequent: Statement,
    val alternate: Statement?,
    val range: List<Int>
) : Statement()

@Serializable
@SerialName("ForStatement")
data class ForStatement(
    val init: VariableDeclaration?,
    val test: Expression?,
    val update: Expression?,
    val body: Statement,
    val range: List<Int>
) : Statement()

@Serializable
@SerialName("WhileStatement")
data class WhileStatement(
    val test: Expression,
    val body: Statement,
    val range: List<Int>
) : Statement()

@Serializable
@SerialName("FunctionDeclaration")
data class FunctionDeclaration(
    val id: Identifier,
    val params: List<Identifier>,
    val body: BlockStatement,
    val range: List<Int>
) : Statement()

@Serializable
@SerialName("ReturnStatement")
data class ReturnStatement(
    val argument: Expression?, // Can be nullable, as return statements might not always return a value
    val range: List<Int>
) : Statement()





fun main() {
    val json = Json {
        prettyPrint = true
        classDiscriminator = "type"
        ignoreUnknownKeys = true
    }

    // Example JSON string
    val jsonString = """
    {
  "type": "Program",
  "body": [
    {
      "type": "FunctionDeclaration",
      "id": {
        "type": "Identifier",
        "name": "fib",
        "range": [
          9,
          12
        ],
        "loc": {
          "start": {
            "line": 1,
            "column": 9
          },
          "end": {
            "line": 1,
            "column": 12
          }
        }
      },
      "params": [
        {
          "type": "Identifier",
          "name": "n",
          "range": [
            13,
            14
          ],
          "loc": {
            "start": {
              "line": 1,
              "column": 13
            },
            "end": {
              "line": 1,
              "column": 14
            }
          }
        }
      ],
      "body": {
        "type": "BlockStatement",
        "body": [
          {
            "type": "IfStatement",
            "test": {
              "type": "BinaryExpression",
              "operator": "<=",
              "left": {
                "type": "Identifier",
                "name": "n",
                "range": [
                  26,
                  27
                ],
                "loc": {
                  "start": {
                    "line": 2,
                    "column": 8
                  },
                  "end": {
                    "line": 2,
                    "column": 9
                  }
                }
              },
              "right": {
                "type": "Literal",
                "value": 2,
                "raw": "2",
                "range": [
                  31,
                  32
                ],
                "loc": {
                  "start": {
                    "line": 2,
                    "column": 13
                  },
                  "end": {
                    "line": 2,
                    "column": 14
                  }
                }
              },
              "range": [
                26,
                32
              ],
              "loc": {
                "start": {
                  "line": 2,
                  "column": 8
                },
                "end": {
                  "line": 2,
                  "column": 14
                }
              }
            },
            "consequent": {
              "type": "ReturnStatement",
              "argument": {
                "type": "Literal",
                "value": 1,
                "raw": "1",
                "range": [
                  41,
                  42
                ],
                "loc": {
                  "start": {
                    "line": 2,
                    "column": 23
                  },
                  "end": {
                    "line": 2,
                    "column": 24
                  }
                }
              },
              "range": [
                34,
                43
              ],
              "loc": {
                "start": {
                  "line": 2,
                  "column": 16
                },
                "end": {
                  "line": 2,
                  "column": 25
                }
              }
            },
            "alternate": null,
            "range": [
              22,
              43
            ],
            "loc": {
              "start": {
                "line": 2,
                "column": 4
              },
              "end": {
                "line": 2,
                "column": 25
              }
            }
          },
          {
            "type": "ReturnStatement",
            "argument": {
              "type": "BinaryExpression",
              "operator": "+",
              "left": {
                "type": "CallExpression",
                "callee": {
                  "type": "Identifier",
                  "name": "fib",
                  "range": [
                    55,
                    58
                  ],
                  "loc": {
                    "start": {
                      "line": 3,
                      "column": 11
                    },
                    "end": {
                      "line": 3,
                      "column": 14
                    }
                  }
                },
                "arguments": [
                  {
                    "type": "BinaryExpression",
                    "operator": "-",
                    "left": {
                      "type": "Identifier",
                      "name": "n",
                      "range": [
                        59,
                        60
                      ],
                      "loc": {
                        "start": {
                          "line": 3,
                          "column": 15
                        },
                        "end": {
                          "line": 3,
                          "column": 16
                        }
                      }
                    },
                    "right": {
                      "type": "Literal",
                      "value": 1,
                      "raw": "1",
                      "range": [
                        63,
                        64
                      ],
                      "loc": {
                        "start": {
                          "line": 3,
                          "column": 19
                        },
                        "end": {
                          "line": 3,
                          "column": 20
                        }
                      }
                    },
                    "range": [
                      59,
                      64
                    ],
                    "loc": {
                      "start": {
                        "line": 3,
                        "column": 15
                      },
                      "end": {
                        "line": 3,
                        "column": 20
                      }
                    }
                  }
                ],
                "range": [
                  55,
                  65
                ],
                "loc": {
                  "start": {
                    "line": 3,
                    "column": 11
                  },
                  "end": {
                    "line": 3,
                    "column": 21
                  }
                }
              },
              "right": {
                "type": "CallExpression",
                "callee": {
                  "type": "Identifier",
                  "name": "fib",
                  "range": [
                    68,
                    71
                  ],
                  "loc": {
                    "start": {
                      "line": 3,
                      "column": 24
                    },
                    "end": {
                      "line": 3,
                      "column": 27
                    }
                  }
                },
                "arguments": [
                  {
                    "type": "BinaryExpression",
                    "operator": "-",
                    "left": {
                      "type": "Identifier",
                      "name": "n",
                      "range": [
                        72,
                        73
                      ],
                      "loc": {
                        "start": {
                          "line": 3,
                          "column": 28
                        },
                        "end": {
                          "line": 3,
                          "column": 29
                        }
                      }
                    },
                    "right": {
                      "type": "Literal",
                      "value": 2,
                      "raw": "2",
                      "range": [
                        76,
                        77
                      ],
                      "loc": {
                        "start": {
                          "line": 3,
                          "column": 32
                        },
                        "end": {
                          "line": 3,
                          "column": 33
                        }
                      }
                    },
                    "range": [
                      72,
                      77
                    ],
                    "loc": {
                      "start": {
                        "line": 3,
                        "column": 28
                      },
                      "end": {
                        "line": 3,
                        "column": 33
                      }
                    }
                  }
                ],
                "range": [
                  68,
                  78
                ],
                "loc": {
                  "start": {
                    "line": 3,
                    "column": 24
                  },
                  "end": {
                    "line": 3,
                    "column": 34
                  }
                }
              },
              "range": [
                55,
                78
              ],
              "loc": {
                "start": {
                  "line": 3,
                  "column": 11
                },
                "end": {
                  "line": 3,
                  "column": 34
                }
              }
            },
            "range": [
              48,
              79
            ],
            "loc": {
              "start": {
                "line": 3,
                "column": 4
              },
              "end": {
                "line": 3,
                "column": 35
              }
            }
          }
        ],
        "range": [
          16,
          81
        ],
        "loc": {
          "start": {
            "line": 1,
            "column": 16
          },
          "end": {
            "line": 4,
            "column": 1
          }
        }
      },
      "generator": false,
      "expression": false,
      "async": false,
      "range": [
        0,
        81
      ],
      "loc": {
        "start": {
          "line": 1,
          "column": 0
        },
        "end": {
          "line": 4,
          "column": 1
        }
      }
    }
  ],
  "sourceType": "script",
  "range": [
    0,
    81
  ],
  "loc": {
    "start": {
      "line": 1,
      "column": 0
    },
    "end": {
      "line": 4,
      "column": 1
    }
  }
}
"""

    // Deserialize
    val program = json.decodeFromString<Program>(jsonString)
    println(program)

    // Serialize back to JSON
    val serializedJson = json.encodeToString(program)
    println(serializedJson)

    val program2 = json.decodeFromString<Program>(serializedJson)
    println(program == program2)

    println()
}

"""
JSON Schemas for MCP Tools
Defines the input/output schemas for web.search and rag.search
"""

from typing import Any

# Tool definitions following MCP specification
TOOL_SCHEMAS = {
    "web.search": {
        "name": "web.search",
        "description": "Search the web for product information, prices, and availability. "
                       "Use this for real-time pricing, current availability checks, or when the user asks about 'current', 'now', or 'latest' information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding products (e.g., 'board game for family under $20')"
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters to narrow search results",
                    "properties": {
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price in USD"
                        },
                        "min_price": {
                            "type": "number", 
                            "description": "Minimum price in USD"
                        },
                        "brand": {
                            "type": "string",
                            "description": "Filter by brand name"
                        },
                        "availability": {
                            "type": "string",
                            "enum": ["in_stock", "any"],
                            "description": "Filter by availability status"
                        }
                    }
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        },
        "outputSchema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Product title"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to the product page"
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Brief description or excerpt"
                    },
                    "price": {
                        "type": "number",
                        "description": "Current price in USD (if available)"
                    },
                    "availability": {
                        "type": "string",
                        "description": "Availability status (e.g., 'In Stock', 'Out of Stock')"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source website name"
                    }
                },
                "required": ["title", "url", "snippet"]
            }
        }
    },
    "rag.search": {
        "name": "rag.search",
        "description": "Search the private Amazon Product Dataset 2020 catalog for product recommendations. "
                       "The catalog contains Games & Accessories products including board games, card games, dice games, and gaming accessories. "
                       "Use this for detailed product information, ratings, and grounded recommendations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing the product you're looking for"
                },
                "budget": {
                    "type": "number",
                    "description": "Maximum budget in USD"
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters for metadata-based filtering",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Product category (e.g., 'Board Games', 'Card Games', 'Game Accessories')"
                        },
                        "brand": {
                            "type": "string",
                            "description": "Filter by brand name"
                        },
                        "min_rating": {
                            "type": "number",
                            "description": "Minimum star rating (1-5)",
                            "minimum": 1,
                            "maximum": 5
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price in USD"
                        },
                        "min_price": {
                            "type": "number",
                            "description": "Minimum price in USD"
                        }
                    }
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 20)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Whether to apply LLM-based reranking for better relevance",
                    "default": False
                }
            },
            "required": ["query"]
        },
        "outputSchema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Product SKU/ID (hash identifier)"
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID for citation purposes"
                    },
                    "title": {
                        "type": "string",
                        "description": "Product title"
                    },
                    "price": {
                        "type": "number",
                        "description": "Price in USD"
                    },
                    "rating": {
                        "type": "number",
                        "description": "Average star rating (may be null)"
                    },
                    "brand": {
                        "type": "string",
                        "description": "Brand name"
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category path"
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Product features list"
                    },
                    "product_url": {
                        "type": "string",
                        "description": "Amazon product URL"
                    },
                    "relevance_score": {
                        "type": "number",
                        "description": "Relevance score from vector search (0-1)"
                    }
                },
                "required": ["sku", "doc_id", "title", "price"]
            }
        }
    }
}


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """
    Get the schema for a specific tool.
    
    Args:
        tool_name: Name of the tool (web.search or rag.search)
        
    Returns:
        Tool schema dict or None if not found
    """
    return TOOL_SCHEMAS.get(tool_name)


def get_tools_list() -> list[dict[str, Any]]:
    """
    Get list of all available tools for MCP tools/list response.
    
    Returns:
        List of tool definitions with name, description, and inputSchema
    """
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "inputSchema": schema["inputSchema"]
        }
        for schema in TOOL_SCHEMAS.values()
    ]


# Example payloads for documentation and testing
EXAMPLE_PAYLOADS = {
    "web.search": {
        "request": {
            "query": "board game cooperative family under $30",
            "filters": {
                "max_price": 30,
                "availability": "in_stock"
            },
            "num_results": 5
        },
        "response": [
            {
                "title": "Pandemic Board Game - Cooperative Strategy",
                "url": "https://amazon.com/dp/B00A2HD40E",
                "snippet": "Award-winning cooperative board game where players work together to save the world from diseases...",
                "price": 29.99,
                "availability": "In Stock",
                "source": "amazon.com"
            },
            {
                "title": "Forbidden Island Board Game",
                "url": "https://amazon.com/dp/B003D7F4YY",
                "snippet": "Cooperative strategy game for 2-4 players. Work together to capture treasures...",
                "price": 19.99,
                "availability": "In Stock",
                "source": "amazon.com"
            }
        ]
    },
    "rag.search": {
        "request": {
            "query": "cooperative board game for 4 players",
            "budget": 30,
            "filters": {
                "category": "Board Games",
                "min_rating": 4.0
            },
            "num_results": 5,
            "rerank": True
        },
        "response": [
            {
                "sku": "43cf83baf4862cb9f887c7dd08650162",
                "doc_id": "43cf83baf4862cb9f887c7dd08650162",
                "title": "Herbaceous",
                "price": 14.01,
                "rating": None,
                "brand": "Herbaceous",
                "category": "Toys & Games | Games & Accessories | Board Games",
                "features": [
                    "For 1-4 Players. Ages 8+",
                    "20 minute playing time",
                    "Simple to teach and learn"
                ],
                "product_url": "https://www.amazon.com/Pencil-First-Games-LLC-pfx500/dp/B01N1L34R9",
                "relevance_score": 0.85
            }
        ]
    }
}

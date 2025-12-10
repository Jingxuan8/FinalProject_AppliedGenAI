# MCP Tool Prompts

This document contains all prompts used for tool-related operations in the MCP server.

## Tool Discovery (tools/list)

The server exposes two tools with the following descriptions:

### web.search
```
Search the web for product information, prices, and availability. 
Use this for real-time pricing, current availability checks, or when 
the user asks about 'current', 'now', or 'latest' information.
```

### rag.search
```
Search the private Amazon Product Dataset 2020 catalog for product recommendations. 
Use this for detailed product information, ratings, reviews, and grounded recommendations.
The catalog contains Games & Accessories products including board games, card games, 
dice games, and gaming accessories.
```

## Planner Rules for Tool Selection

When deciding which tool to use:

1. **Prefer rag.search for facts**: Use the private catalog first for product details, ratings, features, and specifications.

2. **Use web.search for real-time data**: If the user asks about:
   - "current price"
   - "now"
   - "latest"
   - "availability today"
   - "in stock"
   Then call web.search to get live information.

3. **Combine both for comparison**: When the user wants to compare catalog prices with current market prices, call both tools and reconcile results.

## Reranking Prompt (rag.search)

Used when `rerank=true` is specified:

```
Given the user query: "{query}"

Rank these products from most to least relevant (return only the numbers in order, comma-separated):

{products_text}

Most relevant to least relevant:
```

## Tool Call Examples

### web.search Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "web.search",
    "arguments": {
      "query": "board game for family under $20",
      "filters": {
        "max_price": 20,
        "availability": "in_stock"
      },
      "num_results": 5
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"title\": \"Herbaceous Board Game\", \"url\": \"https://amazon.com/dp/B01N1L34R9\", \"snippet\": \"Simple to teach and learn; start in a few minutes and play in twenty...\", \"price\": 14.01, \"availability\": \"In Stock\", \"source\": \"amazon.com\"}]"
      }
    ],
    "isError": false
  }
}
```

### rag.search Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "rag.search",
    "arguments": {
      "query": "cooperative board game for 4 players",
      "budget": 30,
      "filters": {
        "category": "Board Games",
        "min_rating": 4.0
      },
      "num_results": 5,
      "rerank": true
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"sku\": \"43cf83baf4862cb9f887c7dd08650162\", \"doc_id\": \"43cf83baf4862cb9f887c7dd08650162\", \"title\": \"Herbaceous\", \"price\": 14.01, \"rating\": null, \"brand\": \"Herbaceous\", \"category\": \"Toys & Games | Games & Accessories | Board Games\", \"features\": [\"For 1-4 Players. Ages 8+\", \"20 minute playing time\", \"Simple to teach and learn\"], \"relevance_score\": 0.85, \"product_url\": \"https://www.amazon.com/Pencil-First-Games-LLC-pfx500/dp/B01N1L34R9\"}]"
      }
    ],
    "isError": false
  }
}
```

## Data Schema

The private catalog (games_accessories) contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique product identifier (hash) |
| title | string | Product title |
| brand | string | Brand name |
| category | string | Category path (e.g., "Toys & Games \| Games & Accessories \| Board Games") |
| price | float | Price in USD |
| rating | float | Star rating (may be null) |
| features | string | Product features and specifications |
| weight_kg | float | Product weight in kg |
| price_per_kg | float | Normalized price per kg |
| product_url | string | Amazon product URL |

## Conflict Reconciliation

When results from rag.search and web.search differ, reconcile by:

1. **Match by URL/ASIN**: If available, use product URLs or ASIN identifiers to match records
2. **Match by title similarity**: Use fuzzy matching on product titles
3. **Match by brand + title**: Combine brand and title keywords for matching
4. **Flag discrepancies**: If prices differ significantly (>20%), flag for user attention
5. **Prefer catalog data for static info**: Use rag.search for ratings, features, specifications
6. **Prefer web data for dynamic info**: Use web.search for current price and availability

## Category Examples

The catalog covers various game categories:

- `Toys & Games | Games & Accessories` - General games
- `Toys & Games | Games & Accessories | Board Games` - Board games
- `Toys & Games | Games & Accessories | Card Games` - Card games
- `Toys & Games | Games & Accessories | Card Games | Standard Playing Card Decks` - Playing cards
- `Toys & Games | Games & Accessories | Game Accessories` - Gaming accessories
- `Toys & Games | Games & Accessories | Game Accessories | Standard Game Dice` - Dice

import requests
import json
import time
import pandas as pd

# RPC Configuration
rpc_user = 'username'
rpc_password = 'password'
rpc_host = '127.0.0.1'
rpc_port = '45453'

# start and end blocks
start_block_hash = "0000000082cc3f256c36696a7734e8c14fcdf49e95017cddc2ae46d19f3493ca"
end_block_hash = "00000000499ec5b20a55f6327229c190ea3db01c499ef6a0290168b4635f0bb5"
end_block_height = 2523959

# prepare headers
headers = {
    'Content-Type': 'application/json',
}

# Makes a RPC call to the given method with parameters.
def rpc_call(method, params=[]):
  payload = {
      "method": method,
      "params": params,
      "jsonrpc": "2.0",
      "id": "test"
  }
  response = requests.post(f'http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}/', headers=headers, data=json.dumps(payload))
  return response.json()['result']

# Check if a transaction is a coinbase transaction.
def is_coinbase(transaction):  
  return "vin" in transaction and len(transaction["vin"]) > 0 and "coinbase" in transaction["vin"][0]

# Processes a list of transaction IDs to extract relevant data.
def process_transactions(tx_ids):
  transactions = []
  coinbase_addresses = []

  for tx_id in tx_ids:
      raw_transaction = rpc_call('getrawtransaction', [tx_id])
      decoded_transaction = rpc_call('decoderawtransaction', [raw_transaction])
      
      if is_coinbase(decoded_transaction):
          coinbase_vout = decoded_transaction.get("vout", [])
          for vout in coinbase_vout:
              scriptPubKey = vout.get("scriptPubKey", {})
              addresses = scriptPubKey.get("addresses", [])
              coinbase_addresses.extend(addresses)
          continue  # Skip adding this to the transaction list
      
      transaction = {
          "txid": decoded_transaction["txid"],
          "inputs": len(decoded_transaction["vShieldedSpend"]),
          "outputs": len(decoded_transaction["vShieldedOutput"]),
          "bulk": len(decoded_transaction["vShieldedOutput"]) >= 50
      }
      
      transactions.append(transaction)
  
  return transactions, coinbase_addresses

# Retrieves block data for the given block hash.
def get_block_data(block_hash):
  return rpc_call('getblock', [block_hash])

# Format time in seconds to a hours:minutes:seconds format.
def format_time(seconds):
  hours, remainder = divmod(seconds, 3600)
  minutes, seconds = divmod(remainder, 60)
  
  if hours:
      return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
  elif minutes:
      return f"{int(minutes)}m {int(seconds)}s"
  else:
      return f"{int(seconds)}s"


def main(): 
  last_block_time = 0
  start_time = time.time()
  current_block_hash = start_block_hash

  # Initialize an empty dataframe with columns
  columns = ["size", "height", "time", "difficulty", "transaction", 
             "coinbase", "transactions", "elapsed_time", "fastblock", "includes_bulk"]
  df = pd.DataFrame(columns=columns)
  
  while current_block_hash != end_block_hash:
      block_data = get_block_data(current_block_hash)

      transactions, coinbase_addresses = process_transactions(block_data["tx"])
      elapsed_time = block_data["time"] - last_block_time
      
      data = {
          "size": block_data["size"],
          "height": block_data["height"],
          "time": block_data["time"],
          "difficulty": block_data["difficulty"],
          "transaction": len(block_data["tx"]),
          "coinbase": str(coinbase_addresses),
          "transactions": str(transactions),
          "elapsed_time": elapsed_time,
          "fastblock": elapsed_time < 60,
          "includes_bulk": any(tx["bulk"] for tx in transactions)
      }

      # Append data to dataframe
      df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
      
      last_block_time = block_data["time"]
      elapsed_time_script = time.time() - start_time
      blocks_processed = block_data['height'] - get_block_data(start_block_hash)['height'] + 1
      blocks_remaining = end_block_height - block_data['height']
      average_time_per_block = elapsed_time_script / blocks_processed
      estimated_time_remaining = blocks_remaining * average_time_per_block
      
      print(f"height {block_data['height']} | elapsed: {format_time(elapsed_time_script)} | block {blocks_processed} (remaining: {blocks_remaining}) | est time remaining: {format_time(estimated_time_remaining)}")  

      # Update the current_block_hash for the next loop
      current_block_hash = block_data["nextblockhash"]
      
  # Save the dataframe to a Parquet file at the end
  df.to_parquet('block_data.parquet', index=False)

if __name__ == '__main__':
  main()
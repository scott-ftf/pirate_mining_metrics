import os
import sys
import ast
import shutil
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby, count
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from PIL import Image as pil_image

# config
output_path = "outputs/"


# Dictionary of known mining pool addresses
known_addresses = {
    "coolmine_main":  'RTM2Aw6jiSrePbxZNpfFqz4bDpCcMECMiK',
    "coolmine_solo":  'RSiVR1jAnu95MJMdrZDLhsQacwAJ6aUmd9',
    "solopool.org":   'RKE8ouuU2xJKmYNXNj9u9AAX4hxXY32fv3',
    "zergpool":       'RAwQ7QzRymiFDrY1csXpAYEThLBGCpV235',
    "mining-dutch":   'RXgVgBaQ1HwQmNiYu9EBoX9CFG6sDuxBPS',
    "piratepool.io":  'RD5PhyAUhapsvj5ps2cCHozsXZfQSvDdrZ', #marketing
    "piratepool.io":  'RAzq6y7dsUKgfuzNjpzyGiuFzvrwuDheQw', #explorer and bootstrap
    "piratepool.io":  'RKnDd52zJJVtdLNrsLXnh926ojeuToFGiG', #pool infastructure
    "piratepool.io":  'RRL95hu7Pfc4M5uzGL47CQ2rB2rLdpdreg'  #miner payouts
}


# Check if the directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Format time in seconds to a hours:minutes:seconds format
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)    
    if hours:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"
    

# Load data from a parquet file and return the DataFrame
def load_data():
  if len(sys.argv) < 2:
    print("Please provide the path to the parquet file as an argument.")
    sys.exit()
  file_path = sys.argv[1]

  df = pd.read_parquet(file_path)
  return df


# Calculate and return general statistics
def calculate_general_stats(df):
  total_blocks = len(df)
  df['time'] = pd.to_datetime(df['time'], unit='s') 
  start_date = df['time'].min()
  end_date = df['time'].max()
  total_time = (end_date - start_date).total_seconds()
  expected_time = total_blocks * 60

  general_stats = {
    "start_date": start_date,
    "end_date": end_date,
    "total_time": total_time,
    "avg_time_per_block": total_time / total_blocks,
    "expected_time": expected_time,
    "time_difference": total_time - expected_time,
    "total_blocks": total_blocks
  }

  return general_stats


# Calculate and return block speed statistics
def calculate_speed_stats(df, stats):
  speed_stats = {
    "percent_fast_blocks": df[df['elapsed_time'] < 60].shape[0] / stats["total_blocks"] * 100,
    "median_elapsed_fastblock": df[df['fastblock']]['elapsed_time'].median(),
    "median_elapsed_not_fastblock": df[~df['fastblock']]['elapsed_time'].median()     
  }
  return speed_stats

   
# Calculate and return extremes statistics
def calculate_extremes_stats(df):

  def sum_outputs(transactions_string):
    transactions = eval(transactions_string)
    return sum(tx["outputs"] for tx in transactions)

  df['sum_outputs'] = df['transactions'].apply(sum_outputs)
  max_transactions_block = df['transaction'].idxmax()
  max_outputs_block = df['sum_outputs'].idxmax()

  extremes_stats = {
    "top_10_blocks_by_size": df.nlargest(10, 'size')['height'].tolist(),
    "max_transactions_block": max_transactions_block,
    "max_transaction_height": df['height'][max_transactions_block],
    "max_outputs_block": max_outputs_block,
    "max_outputs_height": df['height'][max_outputs_block] 
  }

  return extremes_stats


# Calculate and return difficulty statistics
def calculate_difficulty_stats(df):
  difficulty_stats = {
    "median_difficulty": df['difficulty'].median(),
    "median_difficulty_fastblock": df[df['fastblock']]['difficulty'].median(),
    "median_difficulty_not_fastblock": df[~df['fastblock']]['difficulty'].median() 
  }
  return difficulty_stats


# Prepare coinbase address data frame
def create_coinbase_df(df):   
  coinbase_stats = []
  unique_coinbases = df['coinbase'].unique()

  # Reverse the dictionary for easy lookups
  address_to_name = {v: k for k, v in known_addresses.items()}

  for address in unique_coinbases:
    # Convert the string to a list
    address_list = ast.literal_eval(address)

    address_data = df[df['coinbase'] == address]
    heights = address_data['height'].tolist()
    miner_name = address_to_name.get(address_list[0], "unknown")

    coinbase_stat = {
      "address": address_list[0],
      "miner_name": miner_name,
      "blocks_mined": len(address_data),
      "median_block_time": address_data['elapsed_time'].median(),
      "max_sequential": max([len(list(g)) for k, g in groupby(heights, lambda x, c=count(): next(c)-x)]),
      "bulk_blocks": address_data['includes_bulk'].sum(),
      "bulk_blocks_%": "{:.2f}%".format(address_data['includes_bulk'].mean() * 100),
      "fast_blocks": address_data['fastblock'].sum(),
      "fast_blocks_%": "{:.2f}%".format(address_data['fastblock'].mean() * 100)
    }
    
    # Filter out addresses with fewer than 100 blocks mined
    if coinbase_stat['blocks_mined'] >= 100:
      coinbase_stats.append(coinbase_stat)

  # Convert the list of dictionaries to a pandas DataFrame
  coinbase_df = pd.DataFrame(coinbase_stats)

  # Sort the DataFrame by 'blocks_mined' in descending order
  coinbase_df = coinbase_df.sort_values(by='blocks_mined', ascending=False)

  return coinbase_df


# save stats in text file
def save_stats(df, stats):
  s = stats

  # Open the file for writing
  with open(f"{output_path}statistics.txt", "w") as file:
    # Write the report to the file
    file.write("\n\nGeneral Statistics\n")
    file.write("-----------------------------------\n")
    file.write(f"Start block: {df['height'].iloc[0]}\n")
    file.write(f"Start Date (UTC): {s['start_date']}\n")
    file.write(f"End block: {df['height'].iloc[-1]}\n")
    file.write(f"End Date (UTC): {s['end_date']}\n")

    file.write(f"\nTotal blocks: {s['total_blocks']}\n")
    file.write(f"Expected total time: {format_time(s['expected_time'])}\n")
    file.write(f"Actual total time: {format_time(s['total_time'])}\n")
    file.write(f"Average time per block: {s['avg_time_per_block']:.2f} seconds\n")
    file.write(f"Total Time difference (actual vs expected): {format_time(s['time_difference'])}\n")

    file.write(f"\nPercentage of fast blocks: {s['percent_fast_blocks']:.2f}%\n")
    file.write(f"Median elapsed time for fast blocks: {s['median_elapsed_fastblock']} seconds\n")
    file.write(f"Median elapsed time for non-fast blocks: {s['median_elapsed_not_fastblock']} seconds\n")

    file.write(f"\nTop 10 blocks by size: {s['top_10_blocks_by_size']}\n")
    file.write(f"Block with maximum transactions: {s['max_transaction_height']}\n")
    file.write(f"Block with maximum outputs: {s['max_outputs_height']}\n")

    file.write(f"\nMedian block difficulty: {s['median_difficulty']}\n")
    file.write(f"Median difficulty for fast blocks: {s['median_difficulty_fastblock']}\n")
    file.write(f"Median difficulty for non-fast blocks: {s['median_difficulty_not_fastblock']}\n")


# Transaction Density Analysis
def transaction_density_chart(df):
  plt.figure(figsize=(12, 6))
  df.set_index('time')['transaction'].plot()
  plt.title("Transaction Density Over Time")
  plt.xlabel("Date")
  plt.ylabel("Number of Transactions per Block")
  plt.tight_layout()
  plt.savefig(output_path + 'transaction_density_over_time.png')
  plt.close()

# Frequency Analysis
def frequency_chart(df):
  plt.figure(figsize=(12, 6))
  block_size_frequency = df['size'].value_counts().head(10)
  block_size_frequency.plot(kind='barh')
  plt.title("Top 10 Block Sizes by Frequency")
  plt.xlabel("Number of Occurrences")
  plt.ylabel("Block Size")
  plt.tight_layout()
  plt.savefig(output_path + 'block_size_frequency.png')
  plt.close()

# Correlation Analysis
def correlation_chart(df):
  correlation = df[['size', 'transaction']].corr()
  # print("Correlation between Block Size and Number of Transactions:", correlation.iloc[0, 1])
  plt.figure(figsize=(12, 6))
  sns.scatterplot(data=df, x='size', y='transaction')
  plt.title("Scatterplot between Block Size and Number of Transactions")
  plt.tight_layout()
  plt.savefig(output_path + 'scatterplot_block_size_transactions.png')
  plt.close()

# Block Size & Transaction Relationship
def size_tx_relationship_chart(df):
  plt.figure(figsize=(12, 6))
  avg_transaction_per_block_size = df.groupby('size')['transaction'].mean().reset_index()
  sns.scatterplot(data=avg_transaction_per_block_size, x='size', y='transaction')
  plt.title("Average Number of Transactions per Block Size")
  plt.tight_layout()
  plt.savefig(output_path + 'avg_transactions_per_block_size.png')
  plt.close()

# Time Series Analysis of Difficulty
def difficulty_chart(df):
  plt.figure(figsize=(12, 6))
  df.sort_values('time').plot(x='time', y='difficulty', legend=False)
  plt.title('Time Series Analysis of Difficulty')
  plt.xlabel('Time')
  plt.ylabel('Difficulty')
  plt.tight_layout()
  plt.savefig(output_path + 'difficulty_over_time.png')
  plt.close()

# Histogram of Transaction Counts
def transaction_histogram_chart(df):
  plt.figure(figsize=(12, 6))
  df['transaction'].hist(bins=50, edgecolor='black')
  plt.title('Histogram of Transaction Counts')
  plt.xlabel('Number of Transactions')
  plt.ylabel('Number of Blocks')
  plt.tight_layout()
  plt.savefig(output_path + 'transaction_histogram.png')
  plt.close()

# Clusters of Similar Blocks using K-Means
def clusters_chart(df):
  features = ['size', 'elapsed_time', 'difficulty', 'transaction']
  X = df[features]

  # Standardizing the features
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Using KMeans for clustering. For example, let's assume 4 clusters.
  kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
  df['cluster'] = kmeans.fit_predict(X_scaled)

  # Reduce dimensionality to 2D using PCA for visualization
  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(X_scaled)
  principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

  # Plotting clusters
  plt.figure(figsize=(12, 6))
  colors = ['red', 'blue', 'green', 'yellow']
  for i, color in enumerate(colors):
    plt.scatter(principal_df[df['cluster'] == i]['PC1'], principal_df[df['cluster'] == i]['PC2'], color=color, s=50, label=f'Cluster {i+1}', alpha=0.6)

  plt.title('Clusters of Similar Blocks')
  plt.xlabel("Principal Component 1 (Size & Transaction emphasis)")
  plt.ylabel("Principal Component 2 (Elapsed Time emphasis)")
  plt.legend()
  plt.tight_layout()
  plt.savefig(output_path + 'block_clusters.png')
  plt.close()


# Generates chart plots for various metrics
def generate_charts(df):
  print("Generating Transactions Density Chart")
  transaction_density_chart(df)

  print("Generating Frequency Chart")
  frequency_chart(df)

  print("Generating Transaction Correlation Chart")
  correlation_chart(df)

  print("Generating Transaction Relationship Chart")
  size_tx_relationship_chart(df)

  print("Generating Difficulty Chart")
  difficulty_chart(df)

  print("Generating Transaction Histogram Chart")
  transaction_histogram_chart(df)

  print("Generating Transaction Cluster Chart")
  clusters_chart(df) 


# Save the coinbase Dataframe as a table in a text file
def save_coinbase_df(coinbase_df):  
  original_stdout = sys.stdout # Redirect the standard output to the file
  with open(output_path + "coinbase.txt", 'w') as f:
    sys.stdout = f
    print(coinbase_df.to_string(index=False))
  sys.stdout = original_stdout


# output the charts and statistics in a pdf report
def create_pdf_report():
  # Define the PDF filename
  pdf_filename = "pirate_miner_metrics_report.pdf"
  
  # Create a new document with landscape page size
  doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))
  
  # Create a list to store the elements to be added to the PDF
  story = []
  
  # Add title to the PDF
  styles = getSampleStyleSheet()
  title = Paragraph("Pirate Chain Miner Metrics Report", styles['Title'])
  story.append(title)

  # Customize the style for the date
  date_style = styles['BodyText'].clone('dateStyle')
  date_style.alignment = 1  # 1 is for CENTER alignment
  date_style.fontSize = 16 

  # Generate the current date and time in UTC and format it
  current_utc = datetime.utcnow().strftime('%Y-%m-%d UTC')
  date_paragraph = Paragraph(current_utc, date_style)
  story.append(date_paragraph)
  
  # Add the content of statistics.txt to the PDF
  with open(os.path.join(output_path, "statistics.txt"), 'r') as f:
    content = f.read()

    # Replace newline characters with line break tags for statistics.txt
    content = content.replace('\n', '<br/>')
    para = Paragraph(content, styles['BodyText'])
    story.append(para)
    story.append(PageBreak())

  # Add the content of coinbase.txt as a table to the PDF
  with open(os.path.join(output_path, "coinbase.txt"), 'r') as f:
      lines = f.readlines()
      
      # Convert each line into a list of cells
      data = [line.split() for line in lines]

      # Define a style for the table headers
      header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=1,
        spaceAfter=6
      )

      # Preprocess the headers to replace underscores with spaces and convert them to Paragraph objects
      data[0] = [Paragraph(header.replace("_", " "), header_style) for header in data[0]]

      # Create a table with the data
      colWidths = [
        3.2 * inch,  # address
        1.2 * inch,  # miner name
        0.8 * inch,  # blocks mined
        0.7 * inch,  # median block time
        0.7 * inch,  # max sequential
        0.7 * inch,  # bulk blocks
        0.8 * inch,  # bulk blocks %
        0.8 * inch,  # fast blocks
        0.7 * inch  # fast blocks %
      ]
      table = Table(data, colWidths=colWidths, repeatRows=1)

      # Style the table
      table_style = TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, 0), 'BOTTOM'),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'), 
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12)  
      ])
      table.setStyle(table_style)

      story.append(table)
      story.append(PageBreak())
  

  # Constants for max image size considering page size and margins
  MAX_IMG_WIDTH = 8.4 * inch  # 6.5 inches width
  MAX_IMG_HEIGHT = 10.5 * inch   # 9 inches height

  # Add each of the chart images to the PDF
  for filename in os.listdir(output_path):
    if filename.endswith(".png"):
      # Resize the image for PDF
      img_path = os.path.join(output_path, filename)
      with pil_image.open(img_path) as img:
        img_width, img_height = img.size
        aspect = img_height / float(img_width)

        if aspect > 1:
          # Image is portrait
          new_height = min(MAX_IMG_HEIGHT, img_height)
          new_width = new_height / aspect
        else:
          # Image is landscape or square
          new_width = min(MAX_IMG_WIDTH, img_width)
          new_height = new_width * aspect
        
        # Resize
        img = img.resize((int(new_width), int(new_height)))
        img.save(img_path)
      
      # Add the resized image to the PDF
      img = Image(img_path, width=new_width, height=new_height)
      story.append(img)
      story.append(PageBreak())
  
  # Build the PDF with all the elements
  doc.build(story)


def main():    
  # load the data from the parquet file
  df = load_data()

  # calculate statistics  
  stats = {}
  print("Calculating general statistics") 
  stats.update(calculate_general_stats(df))

  print("Calculating block speed statistics") 
  stats.update(calculate_speed_stats(df, stats))

  print("Calculating extemes statistics") 
  stats.update(calculate_extremes_stats(df))

  print("Calculating difficulty statistics")
  stats.update(calculate_difficulty_stats(df))

  # save the statistics to a text file
  print("saving statistics to statistics.txt") 
  save_stats(df, stats)

  # Create a dataframe of the coinbase addresses and save to a text file
  print("Creating a Data frame of the coinbase adresses")
  coinbase_df = create_coinbase_df(df)
  print("Saving Coinbase DataFrame to coinbase.txt")
  save_coinbase_df(coinbase_df)

  # generate various charts visualizing the data
  print("Generating charts...") 
  generate_charts(df)

  create_pdf_report()
  # fi
  print("Work complete, cleaning up.") 
  # Delete the directory and its contents
  shutil.rmtree(output_path)
  sys.exit(0)

if __name__ == "__main__":
  main()
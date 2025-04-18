import re
import pandas as pd
from collections import Counter
import emoji
from wordcloud import WordCloud
from urlextract import URLExtract
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Load stop words
try:
    with open('stop_words.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
except FileNotFoundError:
    print("Warning: stop_words.txt not found. Using default empty set.")
    stop_words = set()

# Initialize URL extractor
extractor = URLExtract()

def preprocess(data):
    """Parses the WhatsApp chat export text file into a Pandas DataFrame."""
    # Regex pattern attempts to capture different date/time formats
    # Pattern 1: DD/MM/YYYY, HH:MM - User: Message
    pattern1 = r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*)'
    # Pattern 2: MM/DD/YY, HH:MM - User: Message (Common US format)
    pattern2 = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s*[AP]M)\s*-\s*([^:]+):\s*(.*)' # More specific AM/PM
    # Pattern 3: DD/MM/YYYY, HH:MM - Message (No user, system messages or continuation) - Less common export format
    pattern3 = r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})\s*-\s*(.*)' # Simpler pattern if no user is consistently present

    lines = data.strip().split('\n')
    messages_data = []
    current_message = None

    for line in lines:
        parsed = False
        # Try matching patterns
        match1 = re.match(pattern1, line)
        match2 = re.match(pattern2, line)
        match3 = re.match(pattern3, line) # Less likely used first

        match_to_use = None
        date_str, time_str, user_str, message_str = None, None, None, None
        date_format = None

        if match1:
            match_to_use = match1
            date_str, time_str, user_str, message_str = match_to_use.groups()
            date_format = '%d/%m/%Y' if len(date_str.split('/')[-1]) == 4 else '%d/%m/%y'
            parsed = True
        elif match2:
            match_to_use = match2
            date_str, time_str, user_str, message_str = match_to_use.groups()
            date_format = '%m/%d/%y' # Assuming 2-digit year for this format
            # Handle AM/PM if necessary for pd.to_datetime
            time_str = time_str.replace(' AM', '').replace(' PM', '') # Basic handling, might need refinement for 12 AM/PM
            parsed = True
        elif match3 and current_message is None: # Only if no user is found initially
             match_to_use = match3
             date_str, time_str, message_str = match_to_use.groups()
             user_str = "Group Notification" # Assume system message if no user
             date_format = '%d/%m/%Y' if len(date_str.split('/')[-1]) == 4 else '%d/%m/%y'
             parsed = True


        if parsed:
            # If there was a previous message being built, store it
            if current_message:
                messages_data.append(current_message)

            # Start a new message dictionary
            current_message = {
                'raw_date': date_str,
                'raw_time': time_str,
                'user': user_str.strip() if user_str else 'Group Notification',
                'message': message_str.strip(),
                'date_format': date_format
            }
        elif current_message:
            # If the line doesn't match the pattern, append it to the previous message
            current_message['message'] += '\n' + line.strip()

    # Append the last message
    if current_message:
        messages_data.append(current_message)

    if not messages_data:
         # Fallback: try a simpler split if no pattern matched, maybe less structured data?
         # This is a basic fallback and might need significant adjustment based on actual file content
        for line in lines:
             parts = line.split(' - ', 1)
             if len(parts) == 2:
                 date_time_str = parts[0]
                 user_msg_str = parts[1]
                 user_msg_parts = user_msg_str.split(': ', 1)
                 user = user_msg_parts[0] if len(user_msg_parts) == 2 else 'Unknown User'
                 message = user_msg_parts[1] if len(user_msg_parts) == 2 else user_msg_str
                 # Attempt to parse date/time (highly likely to fail without format)
                 messages_data.append({'raw_date': date_time_str, 'raw_time': '', 'user': user, 'message': message, 'date_format': None})

    if not messages_data:
        raise ValueError("Could not parse chat file. Ensure it's a valid WhatsApp export.")


    df = pd.DataFrame(messages_data)

    # Combine raw date and time and parse
    datetime_objects = []
    errors = 0
    for index, row in df.iterrows():
        try:
            if row['date_format'] == '%m/%d/%y': # Handle US format potentially with AM/PM
                 dt_str = f"{row['raw_date']} {row['raw_time']}"
                 # Try parsing with AM/PM if present (basic approach)
                 try:
                    dt_obj = pd.to_datetime(dt_str, format=f"{row['date_format']} %I:%M %p")
                 except ValueError: # Fallback if AM/PM parsing fails or isn't present
                    dt_obj = pd.to_datetime(dt_str, format=f"{row['date_format']} %H:%M")

            elif row['date_format']: # Handle other known formats
                dt_obj = pd.to_datetime(f"{row['raw_date']} {row['raw_time']}", format=f"{row['date_format']} %H:%M")
            else: # If format couldn't be determined
                 dt_obj = pd.to_datetime(row['raw_date']) # Let pandas try guessing
            datetime_objects.append(dt_obj)
        except Exception as e:
            # print(f"Warning: Could not parse date/time for row {index}: {row['raw_date']} {row['raw_time']}. Error: {e}")
            datetime_objects.append(pd.NaT) # Add Not a Time for rows that fail
            errors += 1

    if errors > len(df) * 0.5: # If more than 50% of rows failed parsing
        raise ValueError(f"High number of date parsing errors ({errors}/{len(df)}). Check chat file format.")


    df['date'] = datetime_objects
    df.dropna(subset=['date'], inplace=True) # Drop rows where date parsing failed

    # --- Feature Engineering ---
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['only_date'] = df['date'].dt.date # For daily timeline

    # Period for activity heatmap
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period

    # Drop unnecessary raw columns if needed
    # df.drop(columns=['raw_date', 'raw_time', 'date_format'], inplace=True)

    return df

def fetch_stats(df):
    """Calculates basic statistics."""
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media = df[df['message'] == '<Media omitted>'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))

    return num_messages, len(words), num_media, len(links)

def monthly_timeline(df):
    """Generates monthly message count timeline."""
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + '-' + timeline['year'].astype(str)
    return timeline[['time', 'message']]

def daily_timeline(df):
    """Generates daily message count timeline."""
    daily_timeline_df = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline_df

def weekly_activity(df):
    """Generates weekly activity map (message count per day name)."""
    # Order days correctly
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_activity_df = df.groupby('day_name').count()['message'].reindex(ordered_days).reset_index()
    weekly_activity_df.fillna(0, inplace=True) # Fill NaN if a day has 0 messages
    return weekly_activity_df


def month_activity(df):
    """Generates monthly activity map (message count per month name)."""
     # Order months correctly
    ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_activity_df = df.groupby('month').count()['message'].reindex(ordered_months).reset_index()
    month_activity_df.fillna(0, inplace=True) # Fill NaN if a month has 0 messages
    return month_activity_df


def activity_heatmap(df):
    """Generates heatmap data for day vs period activity."""
    # Order days correctly
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
     # Reindex to ensure correct day order and fill missing days with 0
    user_heatmap = user_heatmap.reindex(ordered_days, fill_value=0)
    return user_heatmap

def most_busy_users(df):
    """Identifies the most busy users."""
    # Exclude 'Group Notification' user
    user_counts = df[df['user'] != 'Group Notification']['user'].value_counts().head()
    user_percentages = round((df[df['user'] != 'Group Notification']['user'].value_counts() / df[df['user'] != 'Group Notification'].shape[0]) * 100, 2)
    user_df = pd.concat([user_counts, user_percentages], axis=1).reset_index()
    user_df.columns = ['User', 'Messages', 'Percentage'] # Rename columns for clarity
    return user_df


def create_wordcloud(df):
    """Generates a WordCloud from messages."""
    # Filter out media messages and group notifications
    temp = df[(df['user'] != 'Group Notification') & (df['message'] != '<Media omitted>')]

    # Function to remove stop words from a message
    def remove_stop_words(message):
        words = []
        for word in message.lower().split():
            if word not in stop_words and word.isalnum(): # Keep only alphanumeric words
                words.append(word)
        return " ".join(words)

    # Apply the function and generate WordCloud
    wc = WordCloud(width=500, height=300, min_font_size=10, background_color='black', colormap='viridis')
    temp['message'] = temp['message'].apply(remove_stop_words)
    all_words = ' '.join(temp['message'])

    if not all_words: # Handle case where there are no words left after filtering
        return None

    df_wc = wc.generate(all_words)
    return df_wc


def most_common_words(df):
    """Finds the most common words used, excluding stop words and media."""
    temp = df[(df['user'] != 'Group Notification') & (df['message'] != '<Media omitted>')]
    words = []
    for message in temp['message']:
        for word in message.lower().split():
            # Basic filtering: alphanumeric and not a stop word
            if word.isalnum() and word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    if most_common_df.empty:
        return pd.DataFrame(columns=['Word', 'Count']) # Return empty DF if no words
    most_common_df.columns = ['Word', 'Count'] # Rename columns
    return most_common_df

def emoji_analysis(df):
    """Performs analysis on emojis used."""
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA]) # Use emoji library data

    if not emojis:
        return pd.DataFrame(columns=['Emoji', 'Count']) # Return empty if no emojis found

    emoji_counts = Counter(emojis).most_common(len(Counter(emojis))) # Get all counts
    emoji_df = pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])
    return emoji_df


# --- Plotting Functions ---

def plot_to_base64(fig):
    """Converts a Matplotlib figure to a Base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor()) # Use figure's facecolor
    plt.close(fig) # Close the figure to free memory
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_monthly_timeline(timeline_df):
    """Plots the monthly timeline."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timeline_df['time'], timeline_df['message'], color='#00bcd4') # Teal color
    plt.xticks(rotation='vertical')
    ax.set_title('Monthly Message Timeline', color='white')
    ax.set_xlabel('Month-Year', color='white')
    ax.set_ylabel('Number of Messages', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.patch.set_facecolor('#1e1e1e') # Dark background for the figure
    ax.set_facecolor('#1e1e1e') # Dark background for the axes
    fig.tight_layout()
    return plot_to_base64(fig)

def plot_daily_timeline(daily_timeline_df):
    """Plots the daily timeline."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_timeline_df['only_date'], daily_timeline_df['message'], color='#8bc34a') # Light green
    plt.xticks(rotation=45)
    ax.set_title('Daily Message Timeline', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Number of Messages', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    fig.tight_layout()
    return plot_to_base64(fig)

def plot_activity_map(activity_df, title, xlabel, color):
    """Plots weekly or monthly activity bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(activity_df.iloc[:, 0], activity_df.iloc[:, 1], color=color) # Use first col for x, second for y
    plt.xticks(rotation=45, ha='right')
    ax.set_title(title, color='white')
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel('Number of Messages', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    fig.tight_layout()
    return plot_to_base64(fig)

def plot_activity_heatmap(heatmap_df):
    """Plots the activity heatmap."""
    if heatmap_df.empty: return None
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(heatmap_df, cmap="viridis", ax=ax, annot=True, fmt=".0f", linewidths=.5, linecolor='gray', cbar_kws={'label': 'Message Count'})
    ax.set_title('Weekly Activity Heatmap (Day vs Time Period)', color='white')
    ax.set_xlabel('Time Period', color='white')
    ax.set_ylabel('Day of Week', color='white')
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white', rotation=0)
    # Set colorbar label color
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(labelcolor='white')
    fig.patch.set_facecolor('#1e1e1e')
    # ax.set_facecolor('#1e1e1e') # Heatmap covers this anyway
    fig.tight_layout()
    return plot_to_base64(fig)

def plot_most_busy_users(users_df):
    """Plots the most busy users bar chart."""
    if users_df.empty: return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(users_df['User'], users_df['Messages'], color='#ff9800') # Orange color
    plt.xticks(rotation=45, ha='right')
    ax.set_title('Most Busy Users (Top 5)', color='white')
    ax.set_xlabel('User', color='white')
    ax.set_ylabel('Number of Messages', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    fig.tight_layout()
    return plot_to_base64(fig)

def plot_wordcloud(wc):
    """Plots the wordcloud."""
    if wc is None: return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off') # Hide axes
    fig.patch.set_facecolor('#1e1e1e')
    # No need to set ax facecolor for imshow
    fig.tight_layout(pad=0)
    return plot_to_base64(fig)

def plot_most_common_words(common_words_df):
    """Plots the most common words bar chart."""
    if common_words_df.empty: return None
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(common_words_df['Word'], common_words_df['Count'], color='#e91e63') # Pink color
    ax.set_title('Top 20 Most Common Words', color='white')
    ax.set_xlabel('Count', color='white')
    ax.set_ylabel('Word', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.invert_yaxis() # Display top word at the top
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    fig.tight_layout()
    return plot_to_base64(fig)

def plot_emoji_analysis(emoji_df):
    """Plots the emoji analysis pie chart."""
    if emoji_df.empty or emoji_df['Count'].sum() == 0: return None

    # Limit to top N emojis for pie chart readability
    top_n = 10
    emoji_df_top = emoji_df.head(top_n)

    # Create labels with emojis and counts
    labels = [f"{row['Emoji']} ({row['Count']})" for index, row in emoji_df_top.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 8)) # Make it slightly larger for pie
    colors = plt.cm.viridis(emoji_df_top['Count'] / float(emoji_df_top['Count'].max())) # Color based on count

    # Explode the largest slice slightly
    explode = [0.1 if i == 0 else 0 for i in range(len(emoji_df_top))]

    wedges, texts, autotexts = ax.pie(emoji_df_top['Count'],
                                      labels=labels,
                                      autopct='%1.1f%%',
                                      startangle=140,
                                      pctdistance=0.85, # Distance of percentage text from center
                                      colors=colors,
                                      wedgeprops={'edgecolor': 'black', 'linewidth': 1}, # Add edge color
                                      explode=explode,
                                      textprops={'color': "white", 'fontsize': 10}) # Style labels

    # Improve label visibility
    plt.setp(autotexts, size=9, weight="bold", color="black") # Percentage text style

    ax.set_title(f'Top {top_n} Emoji Distribution', color='white', fontsize=14)
    fig.patch.set_facecolor('#1e1e1e')
    # ax.set_facecolor('#1e1e1e') # Pie covers this
    fig.tight_layout()
    return plot_to_base64(fig)
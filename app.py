import os
import zipfile
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import helper  
import pandas as pd
import matplotlib
matplotlib.use('Agg') 


app = Flask(__name__)
app.secret_key = 'super secret key' 
app.config['UPLOAD_FOLDER'] = 'uploads' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

ALLOWED_EXTENSIONS = {'txt', 'zip'}

# Create upload folder if it doesn't exist
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_chat():
    if 'chatfile' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['chatfile']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) 
        file_content = ""
        is_zip = filename.lower().endswith('.zip')

        try:
            if is_zip:
                with zipfile.ZipFile(file, 'r') as z:
                    # Look for the standard WhatsApp chat export file name
                    chat_file_name = None
                    for name in z.namelist():
                        if name == '_chat.txt' or name.endswith('.txt'): 
                             chat_file_name = name
                             break
                    if chat_file_name:
                        with z.open(chat_file_name) as f:
                             
                             try:
                                 file_content = f.read().decode('utf-8')
                             except UnicodeDecodeError:
                                 f.seek(0) 
                                 file_content = f.read().decode('latin-1') 
                    else:
                        flash('Could not find a .txt file inside the ZIP archive.', 'error')
                        return redirect(url_for('index'))
            else: 
                 try:
                     file_content = file.read().decode('utf-8')
                 except UnicodeDecodeError:
                     file.seek(0) 
                     file_content = file.read().decode('latin-1')


            # --- Core Analysis ---
            df = helper.preprocess(file_content)

            # 1. Fetch basic stats
            num_messages, num_words, num_media, num_links = helper.fetch_stats(df)

            # 2. Busy Users Analysis (for group chats)
            users = df['user'].unique().tolist()
            most_busy_users_df = None
            most_busy_users_plot = None
            if 'Group Notification' in users:
                users.remove('Group Notification')
            if len(users) > 1: # More than one user indicates a group chat
                most_busy_users_df = helper.most_busy_users(df)
                most_busy_users_plot = helper.plot_most_busy_users(most_busy_users_df)


            # 3. WordCloud
            wordcloud_obj = helper.create_wordcloud(df)
            wordcloud_plot = helper.plot_wordcloud(wordcloud_obj) if wordcloud_obj else None

            # 4. Most Common Words
            common_words_df = helper.most_common_words(df)
            common_words_plot = helper.plot_most_common_words(common_words_df)

            # 5. Emoji Analysis
            emoji_df = helper.emoji_analysis(df)
            emoji_plot = helper.plot_emoji_analysis(emoji_df)
            # Prepare top emojis for display in table/list if needed
            top_emojis_list = emoji_df.head(10).to_dict('records') if not emoji_df.empty else []


            # 6. Timeline Analysis
            monthly_timeline_df = helper.monthly_timeline(df)
            monthly_timeline_plot = helper.plot_monthly_timeline(monthly_timeline_df)

            daily_timeline_df = helper.daily_timeline(df)
            daily_timeline_plot = helper.plot_daily_timeline(daily_timeline_df)

            # 7. Activity Analysis
            weekly_activity_df = helper.weekly_activity(df)
            weekly_activity_plot = helper.plot_activity_map(weekly_activity_df, 'Weekly Activity Distribution', 'Day of Week', '#ffc107') # Amber

            monthly_activity_df = helper.month_activity(df)
            monthly_activity_plot = helper.plot_activity_map(monthly_activity_df, 'Monthly Activity Distribution', 'Month', '#2196f3') # Blue

            # 8. Activity Heatmap
            heatmap_df = helper.activity_heatmap(df)
            heatmap_plot = helper.plot_activity_heatmap(heatmap_df)


            # Pass data to the template
            return render_template('result.html',
                                   num_messages=num_messages,
                                   num_words=num_words,
                                   num_media=num_media,
                                   num_links=num_links,
                                   most_busy_users_plot=most_busy_users_plot,
                                   most_busy_users_data=most_busy_users_df.to_html(classes='styled-table', index=False, justify='center') if most_busy_users_df is not None else "<p>Analysis not applicable for individual chats.</p>",
                                   wordcloud_plot=wordcloud_plot,
                                   common_words_plot=common_words_plot,
                                   common_words_data=common_words_df.to_html(classes='styled-table', index=False, justify='center') if not common_words_df.empty else "<p>No common words found after filtering.</p>",
                                   emoji_plot=emoji_plot,
                                   top_emojis_list=top_emojis_list, # Pass list for potential table display
                                   emoji_data = emoji_df.head(10).to_html(classes='styled-table', index=False, justify='center') if not emoji_df.empty else "<p>No emojis found in the chat.</p>", # Table version
                                   monthly_timeline_plot=monthly_timeline_plot,
                                   daily_timeline_plot=daily_timeline_plot,
                                   weekly_activity_plot=weekly_activity_plot,
                                   monthly_activity_plot=monthly_activity_plot,
                                   heatmap_plot=heatmap_plot,
                                   chat_title=f"Analysis Results for {filename}" # Pass filename as title
                                  )

        except ValueError as e:
            flash(f'Error processing file: {e}', 'error')
            return redirect(url_for('index'))
        except zipfile.BadZipFile:
             flash('Invalid or corrupted ZIP file.', 'error')
             return redirect(url_for('index'))
        except Exception as e:
            # Generic error catch - log this properly in production
            print(f"An unexpected error occurred: {e}") # Log to console for debugging
            flash(f'An unexpected error occurred during analysis. Please ensure the chat file is valid. Error: {e}', 'error')
            return redirect(url_for('index'))

    else:
        flash('Invalid file type. Please upload a .txt or .zip file.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False for production
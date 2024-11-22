import os
import streamlit as st
import pandas as pd
from openai import OpenAI
import pinecone
import nltk
import numpy as np
from textblob import TextBlob
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize APIs and download resources
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone with new syntax
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    st.error("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")
    st.stop()
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

nltk.download('punkt')


# **Update the Index Name**
index_name = 'fsbrands-conflict-index'  # Updated index name
index = pc.Index(index_name)

def parse_transcript(transcript):
    # Split the transcript into lines
    lines = transcript.split('\n')

    segments = []
    i = 0
    while i < len(lines):
        # Look for a timestamp line
        timestamp_pattern = r'\d{2}:\d{2}:\d{2}:\d{2} - \d{2}:\d{2}:\d{2}:\d{2}'
        if re.match(timestamp_pattern, lines[i]):
            timestamp = lines[i].strip()
            i += 1
            # Look for a speaker label
            if i < len(lines) and re.match(r'Speaker \d+', lines[i]):
                speaker = lines[i].strip()
                i += 1
            else:
                speaker = None
            # Collect the text lines until the next timestamp or end of transcript
            text_lines = []
            while i < len(lines) and not re.match(timestamp_pattern, lines[i]):
                text_lines.append(lines[i])
                i += 1
            text = ' '.join(text_lines).strip()
            if text:
                segments.append({'timestamp': timestamp, 'speaker': speaker, 'text': text})
        else:
            i += 1
    return segments

# Set minimum word count
MIN_WORD_COUNT = 5  # Adjust as needed

# App title and description
st.title("Podcast Transcript Conflict Checker")
st.write("Upload a transcript to scan for potential brand conflicts.")

uploaded_file = st.file_uploader("Upload Transcript", type=['txt', 'docx'])

if uploaded_file is not None:
    logging.info("Transcript uploaded.")
    # Read the transcript
    transcript = uploaded_file.read().decode('utf-8')
    logging.info("Transcript read successfully.")

    # Parse the transcript
    segments = parse_transcript(transcript)
    logging.info(f"Transcript parsed into {len(segments)} segments.")

    # Filter and process segments
    processed_segments = []
    for segment in segments:
        timestamp = segment['timestamp']
        speaker = segment['speaker']
        text = segment['text']
        word_count = len(text.split())
        if word_count >= MIN_WORD_COUNT:
            processed_segments.append({'text': text, 'timestamp': timestamp, 'speaker': speaker})
        else:
            logging.info(f"Segment too short, skipping: {text}")
    logging.info(f"Filtered to {len(processed_segments)} segments after removing short ones.")
    st.write(f"Transcript contains {len(processed_segments)} segments after preprocessing.")

    # Generate embeddings for processed segments
    def get_embeddings(text_list):
        try:
            response = client.embeddings.create(
                input=text_list,
                model="text-embedding-ada-002"
            )
            embeddings = [data.embedding for data in response.data]
            logging.info(f"Generated embeddings for {len(text_list)} texts.")
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            return None

    logging.info("Starting to generate embeddings for transcript segments.")
    st.write("Generating embeddings for transcript segments...")
    segment_texts = [s['text'] for s in processed_segments]
    segment_embeddings = []
    batch_size = 50  # Adjust as needed
    total_batches = len(segment_texts) // batch_size + 1
    progress_bar = st.progress(0)

    for batch_num, i in enumerate(range(0, len(segment_texts), batch_size)):
        batch_texts = segment_texts[i:i+batch_size]
        embeddings = get_embeddings(batch_texts)
        if embeddings:
            segment_embeddings.extend(embeddings)
            logging.info(f"Generated embeddings for batch {batch_num + 1}/{total_batches}.")
        else:
            logging.error(f"Failed to generate embeddings for batch {batch_num + 1}.")
        # Update progress bar
        progress = (batch_num + 1) / total_batches
        progress_bar.progress(progress)

    st.write("Generated embeddings for transcript segments.")
    logging.info("Completed generating embeddings for all segments.")

    # Query Pinecone index
    def query_pinecone(embedding, top_k=3):
        try:
            query_response = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            logging.info(f"Queried Pinecone index; retrieved {len(query_response.matches)} matches.")
            return query_response.matches
        except Exception as e:
            logging.error(f"Error querying Pinecone: {e}")
            return []

    logging.info("Starting to query Pinecone for potential conflicts.")
    st.write("Querying Pinecone for potential conflicts...")
    query_progress_bar = st.progress(0)
    potential_conflicts = []
    total_queries = len(segment_embeddings)

    for idx, embedding in enumerate(segment_embeddings):
        matches = query_pinecone(embedding)
        for match in matches:
            similarity_score = match.score
            if similarity_score >= 0.80:
                conflict = {
                    'Text': processed_segments[idx]['text'],
                    'Timestamp': processed_segments[idx]['timestamp'],
                    'Speaker': processed_segments[idx]['speaker'],
                    'Brand': match.metadata['brand'],
                    'Category': match.metadata['category'],
                    'Matched Text': match.metadata['text'],
                    'Similarity Score': similarity_score
                }
                potential_conflicts.append(conflict)
                logging.info(f"Potential conflict found: {conflict}")
        # Update progress bar
        progress = (idx + 1) / total_queries
        query_progress_bar.progress(progress)

    logging.info("Completed querying Pinecone.")
    st.write("Completed querying for potential conflicts.")

    # Apply sentiment analysis
    logging.info("Starting sentiment analysis on potential conflicts.")
    for conflict in potential_conflicts:
        sentiment = TextBlob(conflict['Text']).sentiment.polarity
        conflict['Sentiment Score'] = sentiment
    logging.info("Completed sentiment analysis.")

    # Assign severity
    def assign_severity(similarity, sentiment):
        if sentiment < -0.5 and similarity > 0.80:
            return 'High'
        elif sentiment < 0 and similarity > 0.83:
            return 'Medium'
        else:
            return 'Low'

    for conflict in potential_conflicts:
        conflict['Severity'] = assign_severity(conflict['Similarity Score'], conflict['Sentiment Score'])

    # Prepare data for summarization
    high_medium_conflicts = [conflict for conflict in potential_conflicts if conflict['Severity'] in ['High', 'Medium']]

    def generate_summary(conflicts):
        prompt = "Summarize the following potential conflicts detected in the podcast transcript. Provide a concise summary highlighting the main issues that need to be addressed:\n\n"
        for idx, conflict in enumerate(conflicts, 1):
            prompt += f"{idx}. Timestamp: {conflict['Timestamp']}\n"
            prompt += f"Speaker: {conflict['Speaker']}\n"
            prompt += f"Text: \"{conflict['Text']}\"\n"
            prompt += f"Matched Brand: {conflict['Brand']}\n"
            prompt += f"Category: {conflict['Category']}\n"
            prompt += f"Severity: {conflict['Severity']}\n\n"
        prompt += "Summary:"
        return prompt

    def get_summary_from_openai(prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing and summarizing potential brand conflicts in podcast transcripts."},
                    {"role": "user", "content": prompt}
                ]
            )
            summary = response.choices[0].message.content.strip()
            logging.info("Generated summary from OpenAI.")
            return summary
        except Exception as e:
            logging.error(f"Error generating summary from OpenAI: {e}")
            return None

    # Display summary if there are High or Medium severity conflicts
    if high_medium_conflicts:
        st.subheader("Summary of Potential Conflicts")
        prompt = generate_summary(high_medium_conflicts)
        with st.spinner("Generating summary of High and Medium severity conflicts..."):
            summary = get_summary_from_openai(prompt)
            if summary:
                st.markdown(summary)
            else:
                st.error("Failed to generate summary.")

    # Display results
    if potential_conflicts:
        conflicts_df = pd.DataFrame(potential_conflicts)
        logging.info(f"Found {len(conflicts_df)} potential conflicts.")

        # Define severity order
        severity_order = {'High': 1, 'Medium': 2, 'Low': 3}
        conflicts_df['Severity Rank'] = conflicts_df['Severity'].map(severity_order)

        # Sort the DataFrame
        conflicts_df.sort_values(by=['Severity Rank', 'Similarity Score'], ascending=[True, False], inplace=True)

        st.subheader("Potential Conflicts Detected")
        st.dataframe(conflicts_df[['Timestamp', 'Speaker', 'Brand', 'Category', 'Matched Text', 'Text', 'Similarity Score', 'Sentiment Score', 'Severity']])

        # Download option
        csv = conflicts_df[['Timestamp', 'Speaker', 'Brand', 'Category',
                   'Matched Text', 'Text', 'Similarity Score',
                   'Sentiment Score', 'Severity']].to_csv(index=False)
        st.download_button(
            label="Download Report as CSV",
            data=csv,
            file_name='conflict_report.csv',
            mime='text/csv',
        )

        # Display the highlighted transcript by default
        st.subheader("Highlighted Transcript")
        # Reconstruct the transcript with HTML and tooltips
        reconstructed_transcript = ''
        for segment in segments:
            reconstructed_transcript += f"{segment['timestamp']}\n"
            if segment['speaker']:
                reconstructed_transcript += f"{segment['speaker']}\n"
            reconstructed_transcript += f"{segment['text']}\n\n"

        # Highlight the conflicting segments with tooltips
        highlighted_transcript = reconstructed_transcript
        for conflict in potential_conflicts:
            highlighted_text = conflict['Text']
            if highlighted_text in highlighted_transcript:
                # Create tooltip content with proper formatting
                tooltip = (
                    f"Brand: {conflict['Brand']}\n"
                    f"Category: {conflict['Category']}\n"
                    f"Severity: {conflict['Severity']}\n"
                    f"Similarity: {conflict['Similarity Score']:.2f}\n"
                    f"Sentiment: {conflict['Sentiment Score']:.2f}"
                )

                # Escape HTML and replace newlines with proper spacing
                tooltip = tooltip.replace('"', '&quot;').replace('\n', ' • ')

                # Create HTML with tooltip
                highlight_color = {
                    'High': '#ffb3b3',    # Light red
                    'Medium': '#fff2b3',   # Light yellow
                    'Low': '#e6ffe6'       # Light green
                }[conflict['Severity']]

                highlighted_html = (
                    f'<span class="highlight-tooltip" '
                    f'style="background-color: {highlight_color};" '
                    f'data-tooltip="{tooltip}">'
                    f'{highlighted_text}'
                    f'</span>'
                )

                highlighted_transcript = highlighted_transcript.replace(
                    highlighted_text,
                    highlighted_html
                )

        # Add improved CSS for better tooltip display
        st.markdown("""
            <style>
            .highlight-tooltip {
                position: relative;
                cursor: help;
            }

            .highlight-tooltip:hover::before {
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background-color: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 1000;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }

            .highlight-tooltip:hover::after {
                content: '';
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent;
                border-top-color: rgba(0, 0, 0, 0.8);
                margin-bottom: -12px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display the highlighted transcript with HTML rendering
        st.markdown(highlighted_transcript, unsafe_allow_html=True)
    else:
        st.success("No potential conflicts detected.")
        logging.info("No potential conflicts detected.")
# visualization_generator.py - Chart Generation for RA Dashboard (PP-12, PP-15)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import base64
import io

class RAVisualizationGenerator:
    """Generate interactive visualizations for RA dashboard"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'pain_high': '#d62728',
            'pain_medium': '#ff7f0e',
            'pain_low': '#2ca02c',
            'risk_high': '#dc3545',
            'risk_moderate': '#fd7e14',
            'risk_low': '#20c997',
            'risk_minimal': '#28a745'
        }
        
    def generate_pain_level_chart(self, data: pd.DataFrame, time_range: str = '30d') -> Dict:
        """Generate interactive pain level chart over time (PP-12)"""
        
        if len(data) == 0:
            return self._empty_chart_response("No data available")
        
        # Ensure date column exists
        if 'date' not in data.columns:
            data = data.copy()
            data['date'] = pd.date_range(end=datetime.now(), periods=len(data), freq='D')
        
        # Filter by time range
        data = self._filter_by_time_range(data, time_range)
        
        if len(data) == 0:
            return self._empty_chart_response(f"No data available for {time_range}")
        
        # Create main pain level chart
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Pain Levels Over Time', 'Risk Predictions', 'Weather Correlation'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Main pain trend line
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['pain_history_1d'],
                mode='lines+markers',
                name='Daily Pain Level',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>Pain Level: %{y:.1f}/10<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Pain level zones (background shading)
        fig.add_hrect(y0=1, y1=3, fillcolor=self.color_scheme['pain_low'], 
                     opacity=0.1, layer="below", row=1, col=1)
        fig.add_hrect(y0=3, y1=6, fillcolor=self.color_scheme['pain_medium'], 
                     opacity=0.1, layer="below", row=1, col=1)
        fig.add_hrect(y0=6, y1=10, fillcolor=self.color_scheme['pain_high'], 
                     opacity=0.1, layer="below", row=1, col=1)
        
        # Add 7-day moving average
        if len(data) >= 7:
            data['pain_ma_7d'] = data['pain_history_1d'].rolling(7, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['pain_ma_7d'],
                    mode='lines',
                    name='7-day Average',
                    line=dict(color=self.color_scheme['secondary'], width=2, dash='dash'),
                    hovertemplate='<b>%{x}</b><br>7-day Average: %{y:.1f}/10<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Risk predictions (if available)
        if 'predicted_flare_probability' in data.columns or 'flare_probability' in data.columns:
            risk_col = 'predicted_flare_probability' if 'predicted_flare_probability' in data.columns else 'flare_probability'
            
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data[risk_col] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Flare Risk %',
                    line=dict(color=self.color_scheme['warning'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{x}</b><br>Flare Risk: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Risk level zones
            fig.add_hrect(y0=0, y1=20, fillcolor=self.color_scheme['risk_minimal'], 
                         opacity=0.1, layer="below", row=2, col=1)
            fig.add_hrect(y0=20, y1=40, fillcolor=self.color_scheme['risk_low'], 
                         opacity=0.1, layer="below", row=2, col=1)
            fig.add_hrect(y0=40, y1=70, fillcolor=self.color_scheme['risk_moderate'], 
                         opacity=0.1, layer="below", row=2, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor=self.color_scheme['risk_high'], 
                         opacity=0.1, layer="below", row=2, col=1)
        
        # Weather correlation (temperature as example)
        if 'temperature' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['temperature'],
                    mode='lines',
                    name='Temperature (°C)',
                    line=dict(color=self.color_scheme['info'], width=1.5),
                    yaxis='y3',
                    hovertemplate='<b>%{x}</b><br>Temperature: %{y:.1f}°C<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Pain Level Analysis - Last {time_range}',
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            height=800,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Pain Level (1-10)", row=1, col=1, range=[0, 10])
        fig.update_yaxes(title_text="Risk Probability (%)", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        # Generate analytics summary
        analytics = self._generate_pain_analytics(data)
        
        return {
            'chart_html': fig.to_html(include_plotlyjs='cdn'),
            'chart_json': fig.to_json(),
            'analytics': analytics,
            'data_points': len(data),
            'time_range': time_range
        }
    
    def generate_correlation_matrix(self, correlation_data: Dict) -> Dict:
        """Generate correlation matrix heatmap (PP-14)"""
        
        if not correlation_data or 'correlation_matrix' not in correlation_data:
            return self._empty_chart_response("No correlation data available")
        
        # Extract Pearson correlations
        correlations = correlation_data['correlation_matrix'].get('pearson', {})
        
        if not correlations:
            return self._empty_chart_response("No correlation matrix data")
        
        # Convert to matrix format
        symptoms = list(correlations.keys())
        triggers = list(correlations[symptoms[0]].keys()) if symptoms else []
        
        correlation_matrix = []
        for symptom in symptoms:
            row = []
            for trigger in triggers:
                corr_val = correlations[symptom][trigger].get('correlation', 0)
                row.append(corr_val)
            correlation_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=triggers,
            y=symptoms,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title="Correlation Coefficient",
                titleside="right"
            ),
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Add significance indicators
        for i, symptom in enumerate(symptoms):
            for j, trigger in enumerate(triggers):
                corr_data = correlations[symptom][trigger]
                if corr_data.get('significant', False):
                    fig.add_annotation(
                        x=j, y=i,
                        text="*",
                        showarrow=False,
                        font=dict(color="white", size=16)
                    )
        
        fig.update_layout(
            title=dict(
                text='Symptom-Trigger Correlation Matrix',
                x=0.5,
                font=dict(size=18, color='#2c3e50')
            ),
            xaxis_title="Potential Triggers",
            yaxis_title="Symptoms",
            height=600,
            template='plotly_white'
        )
        
        # Generate insights
        insights = self._generate_correlation_insights(correlation_data)
        
        return {
            'chart_html': fig.to_html(include_plotlyjs='cdn'),
            'chart_json': fig.to_json(),
            'insights': insights,
            'significant_correlations': len([
                1 for symptom in correlations.values() 
                for trigger_data in symptom.values() 
                if trigger_data.get('significant', False)
            ])
        }
    
    def generate_risk_dashboard(self, risk_analytics: Dict) -> Dict:
        """Generate comprehensive risk dashboard (PP-90)"""
        
        if not risk_analytics:
            return self._empty_chart_response("No risk analytics available")
        
        # Create dashboard with multiple components
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Current Risk Level', 
                'Risk Trend (30 days)',
                'Flare Frequency',
                'Seasonal Pattern'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Current Risk Gauge
        current_risk = risk_analytics.get('current_risk_score', 0.5)
        risk_level = risk_analytics.get('risk_level', 'UNKNOWN')
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_risk * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Risk Level: {risk_level}"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_risk_color(current_risk)},
                    'steps': [
                        {'range': [0, 20], 'color': self.color_scheme['risk_minimal']},
                        {'range': [20, 40], 'color': self.color_scheme['risk_low']},
                        {'range': [40, 70], 'color': self.color_scheme['risk_moderate']},
                        {'range': [70, 100], 'color': self.color_scheme['risk_high']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Risk Trend (if available)
        if 'risk_change_7d' in risk_analytics and 'risk_change_30d' in risk_analytics:
            risk_changes = [
                risk_analytics.get('risk_change_7d', 0),
                risk_analytics.get('risk_change_30d', 0)
            ]
            periods = ['7 days', '30 days']
            colors = ['green' if x < 0 else 'red' for x in risk_changes]
            
            fig.add_trace(
                go.Bar(
                    x=periods,
                    y=risk_changes,
                    marker_color=colors,
                    name='Risk Change',
                    hovertemplate='<b>%{x}</b><br>Change: %{y:+.3f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. Flare Frequency
        flare_analysis = risk_analytics.get('flare_frequency', {})
        if flare_analysis:
            monthly_avg = flare_analysis.get('monthly_average', 0)
            fig.add_trace(
                go.Bar(
                    x=['Monthly Average'],
                    y=[monthly_avg],
                    marker_color=self.color_scheme['warning'],
                    name='Flare Frequency',
                    hovertemplate='Monthly Average: %{y:.1f} flares<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Seasonal Pattern
        seasonal_patterns = risk_analytics.get('seasonal_patterns', {})
        monthly_risks = seasonal_patterns.get('monthly_risk_scores', {})
        if monthly_risks:
            months = list(monthly_risks.keys())
            risks = list(monthly_risks.values())
            
            fig.add_trace(
                go.Bar(
                    x=months,
                    y=risks,
                    marker_color=self.color_scheme['info'],
                    name='Seasonal Risk',
                    hovertemplate='<b>%{x}</b><br>Risk Score: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text='Personal Risk Dashboard',
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Generate summary insights
        insights = self._generate_risk_insights(risk_analytics)
        
        return {
            'chart_html': fig.to_html(include_plotlyjs='cdn'),
            'chart_json': fig.to_json(),
            'insights': insights,
            'current_risk_level': risk_level,
            'current_risk_score': current_risk
        }
    
    def generate_comprehensive_history_view(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive symptom history view (PP-15)"""
        
        if len(data) == 0:
            return self._empty_chart_response("No historical data available")
        
        # Create comprehensive timeline
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'Pain Levels & Flare Events',
                'Lifestyle Factors',
                'Weather Conditions',
                'Medication Adherence'
            ),
            vertical_spacing=0.06,
            row_heights=[0.35, 0.25, 0.25, 0.15]
        )
        
        # Ensure date column
        if 'date' not in data.columns:
            data = data.copy()
            data['date'] = pd.date_range(end=datetime.now(), periods=len(data), freq='D')
        
        # 1. Pain levels and flare events
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['pain_history_1d'],
                mode='lines+markers',
                name='Pain Level',
                line=dict(color=self.color_scheme['primary'], width=2),
                hovertemplate='<b>%{x}</b><br>Pain: %{y:.1f}/10<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add flare events if available
        if 'flare_occurred' in data.columns:
            flare_events = data[data['flare_occurred'] == 1]
            if len(flare_events) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=flare_events['date'],
                        y=[9] * len(flare_events),  # Position at top
                        mode='markers',
                        name='Flare Events',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.color_scheme['warning']
                        ),
                        hovertemplate='<b>%{x}</b><br>Flare Event<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 2. Lifestyle factors
        if 'stress_level' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['stress_level'],
                    mode='lines',
                    name='Stress Level',
                    line=dict(color=self.color_scheme['warning'], width=2),
                    hovertemplate='<b>%{x}</b><br>Stress: %{y:.1f}/10<extra></extra>'
                ),
                row=2, col=1
            )
        
        if 'sleep_quality' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['sleep_quality'],
                    mode='lines',
                    name='Sleep Quality',
                    line=dict(color=self.color_scheme['success'], width=2),
                    hovertemplate='<b>%{x}</b><br>Sleep: %{y:.1f}/10<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 3. Weather conditions
        if 'temperature' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['temperature'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color=self.color_scheme['info'], width=1.5),
                    hovertemplate='<b>%{x}</b><br>Temp: %{y:.1f}°C<extra></extra>'
                ),
                row=3, col=1
            )
        
        if 'humidity' in data.columns:
            # Scale humidity to similar range as temperature for visualization
            humidity_scaled = (data['humidity'] - 50) / 2  # Center around 0, scale down
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=humidity_scaled,
                    mode='lines',
                    name='Humidity (scaled)',
                    line=dict(color=self.color_scheme['secondary'], width=1.5),
                    hovertemplate='<b>%{x}</b><br>Humidity: %{customdata:.0f}%<extra></extra>',
                    customdata=data['humidity']
                ),
                row=3, col=1
            )
        
        # 4. Medication adherence
        if 'medication_adherence' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data['medication_adherence'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Medication Adherence',
                    line=dict(color=self.color_scheme['success'], width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{x}</b><br>Adherence: %{y:.1f}%<extra></extra>'
                ),
                row=4, col=1
            )
            
            # Add adherence threshold line
            fig.add_hline(
                y=80, line_dash="dash", line_color="red",
                annotation_text="Target: 80%",
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Comprehensive Symptom History',
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            height=1000,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Pain Level (1-10)", row=1, col=1, range=[0, 10])
        fig.update_yaxes(title_text="Level (1-10)", row=2, col=1, range=[0, 10])
        fig.update_yaxes(title_text="Weather", row=3, col=1)
        fig.update_yaxes(title_text="Adherence (%)", row=4, col=1, range=[0, 100])
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        # Generate historical insights
        insights = self._generate_historical_insights(data)
        
        return {
            'chart_html': fig.to_html(include_plotlyjs='cdn'),
            'chart_json': fig.to_json(),
            'insights': insights,
            'data_points': len(data),
            'date_range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}"
        }
    
    # Helper methods
    def _filter_by_time_range(self, data: pd.DataFrame, time_range: str) -> pd.DataFrame:
        """Filter data by time range"""
        if 'date' not in data.columns:
            return data
        
        data['date'] = pd.to_datetime(data['date'])
        end_date = data['date'].max()
        
        if time_range == '7d':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30d':
            start_date = end_date - timedelta(days=30)
        elif time_range == '90d':
            start_date = end_date - timedelta(days=90)
        elif time_range == '1y':
            start_date = end_date - timedelta(days=365)
        else:
            return data
        
        return data[data['date'] >= start_date]
    
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        if risk_score >= 0.7:
            return self.color_scheme['risk_high']
        elif risk_score >= 0.4:
            return self.color_scheme['risk_moderate']
        elif risk_score >= 0.2:
            return self.color_scheme['risk_low']
        else:
            return self.color_scheme['risk_minimal']
    
    def _empty_chart_response(self, message: str) -> Dict:
        """Return empty chart response"""
        return {
            'chart_html': f'<div style="text-align: center; padding: 50px;">{message}</div>',
            'chart_json': '{}',
            'error': message
        }
    
    def _generate_pain_analytics(self, data: pd.DataFrame) -> Dict:
        """Generate pain analytics summary"""
        if 'pain_history_1d' not in data.columns or len(data) == 0:
            return {}
        
        pain_data = data['pain_history_1d']
        
        return {
            'average_pain': float(pain_data.mean()),
            'max_pain': float(pain_data.max()),
            'min_pain': float(pain_data.min()),
            'pain_trend': 'improving' if len(pain_data) > 1 and pain_data.iloc[-1] < pain_data.iloc[0] else 'stable',
            'high_pain_days': int((pain_data > 6).sum()),
            'low_pain_days': int((pain_data <= 3).sum())
        }
    
    def _generate_correlation_insights(self, correlation_data: Dict) -> List[str]:
        """Generate correlation insights"""
        insights = []
        
        if 'personalized_insights' in correlation_data:
            insights.extend(correlation_data['personalized_insights'])
        
        # Add statistical summary
        strong_correlations = correlation_data.get('strong_correlations', [])
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} statistically significant correlations")
        
        return insights
    
    def _generate_risk_insights(self, risk_analytics: Dict) -> List[str]:
        """Generate risk dashboard insights"""
        insights = []
        
        risk_level = risk_analytics.get('risk_level', 'UNKNOWN')
        trend = risk_analytics.get('trend_direction', 'stable')
        
        if risk_level == 'HIGH':
            insights.append("⚠️ High risk detected - consider consulting healthcare provider")
        elif risk_level == 'MODERATE':
            insights.append("📊 Moderate risk - monitor symptoms closely")
        else:
            insights.append("✅ Low risk - continue current management")
        
        if trend == 'improving':
            insights.append("📈 Risk trend is improving")
        elif trend == 'worsening':
            insights.append("📉 Risk trend shows concern - review recent changes")
        
        return insights
    
    def _generate_historical_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate historical data insights"""
        insights = []
        
        if len(data) < 7:
            insights.append("More data needed for comprehensive analysis")
            return insights
        
        # Pain trend
        if 'pain_history_1d' in data.columns:
            recent_pain = data['pain_history_1d'].tail(7).mean()
            older_pain = data['pain_history_1d'].head(7).mean()
            
            if recent_pain < older_pain - 0.5:
                insights.append("📈 Pain levels are improving over time")
            elif recent_pain > older_pain + 0.5:
                insights.append("📉 Pain levels have increased recently")
            else:
                insights.append("📊 Pain levels are relatively stable")
        
        # Flare frequency
        if 'flare_occurred' in data.columns:
            flare_rate = data['flare_occurred'].mean()
            if flare_rate > 0.2:
                insights.append("⚠️ Frequent flares detected")
            elif flare_rate > 0.1:
                insights.append("📊 Moderate flare frequency")
            else:
                insights.append("✅ Low flare frequency")
        
        # Medication adherence
        if 'medication_adherence' in data.columns:
            avg_adherence = data['medication_adherence'].mean()
            if avg_adherence < 0.7:
                insights.append("💊 Medication adherence below recommended 70%")
            elif avg_adherence > 0.9:
                insights.append("✅ Excellent medication adherence")
        
        return insights


# Example usage and testing
if __name__ == "__main__":
    print("RA Visualization Generator - Example Usage")
    print("="*50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'pain_history_1d': np.random.normal(4, 2, 90).clip(1, 10),
        'temperature': np.random.normal(15, 8, 90),
        'humidity': np.random.normal(60, 20, 90).clip(0, 100),
        'pressure': np.random.normal(1013, 15, 90),
        'stress_level': np.random.normal(5, 2, 90).clip(1, 10),
        'sleep_quality': np.random.normal(6, 2, 90).clip(1, 10),
        'medication_adherence': np.random.beta(8, 2, 90),
        'flare_occurred': np.random.binomial(1, 0.12, 90),
        'predicted_flare_probability': np.random.beta(2, 8, 90)
    })
    
    # Initialize visualizer
    viz_generator = RAVisualizationGenerator()
    
    # Test pain level chart
    print("Generating pain level chart...")
    pain_chart = viz_generator.generate_pain_level_chart(sample_data, '30d')
    print(f"Pain chart generated with {pain_chart['data_points']} data points")
    
    # Test comprehensive history view
    print("Generating comprehensive history view...")
    history_view = viz_generator.generate_comprehensive_history_view(sample_data)
    print(f"History view generated covering {history_view['date_range']}")
    
    # Test risk dashboard
    print("Generating risk dashboard...")
    sample_risk_analytics = {
        'current_risk_score': 0.35,
        'risk_level': 'MODERATE',
        'risk_change_7d': -0.05,
        'risk_change_30d': 0.02,
        'trend_direction': 'improving',
        'flare_frequency': {'monthly_average': 2.5},
        'seasonal_patterns': {
            'monthly_risk_scores': {
                'Jan': 0.4, 'Feb': 0.35, 'Mar': 0.3, 'Apr': 0.25,
                'May': 0.2, 'Jun': 0.15, 'Jul': 0.18, 'Aug': 0.22,
                'Sep': 0.28, 'Oct': 0.35, 'Nov': 0.4, 'Dec': 0.45
            }
        }
    }
    
    risk_dashboard = viz_generator.generate_risk_dashboard(sample_risk_analytics)
    print(f"Risk dashboard generated with {len(risk_dashboard['insights'])} insights")
    
    print("\nAll visualizations generated successfully!")
    print("Charts can be embedded in web applications or saved as HTML files.")
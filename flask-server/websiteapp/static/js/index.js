document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('query-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    const plotlyChart = document.getElementById('plotly-chart');
    const downloadButtons = document.querySelector('.download-buttons');

    queryForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission
        loadingSpinner.classList.remove('d-none'); // Show spinner
        plotlyChart.innerHTML = ''; // Clear previous chart
        downloadButtons.classList.add('d-none'); // Hide download buttons

        const formData = new FormData(queryForm);

        fetch(queryForm.action, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errData => { throw errData; });
            }
            return response.json();
        })
        .then(data => {
            loadingSpinner.classList.add('d-none'); // Hide spinner

            if (data.plotly_json) {
                const plotlyData = JSON.parse(data.plotly_json);
                Plotly.newPlot(plotlyChart, plotlyData.data, plotlyData.layout || {});
                downloadButtons.classList.remove('d-none'); // Show download buttons
            }

            if (data.response_text) {
                // Display response text as an alert
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-info alert-dismissible fade show';
                alertDiv.role = 'alert';
                alertDiv.innerHTML = `
                    ${data.response_text}
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                plotlyChart.appendChild(alertDiv);
            }
        })
        .catch(errorData => {
            loadingSpinner.classList.add('d-none'); // Hide spinner
            // Display error messages
            if (errorData.response_text) {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-danger alert-dismissible fade show';
                alertDiv.role = 'alert';
                alertDiv.innerHTML = `
                    ${errorData.response_text}
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                `;
                plotlyChart.appendChild(alertDiv);
            } else if (errorData.message) {
                alert(`Error: ${errorData.message}`);
            } else {
                alert('An unexpected error occurred.');
            }
        });
    });

    // Download buttons event listeners
    document.getElementById('download-png').addEventListener('click', function() {
        downloadVisualization('png');
    });

    document.getElementById('download-svg').addEventListener('click', function() {
        downloadVisualization('svg');
    });

    document.getElementById('download-pdf').addEventListener('click', function() {
        downloadVisualization('pdf');
    });

    function downloadVisualization(format) {
        fetch('/download_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ format: format })
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => { throw new Error(text); });
            }
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `plot.${format}`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            alert(`Download failed: ${error.message}`);
        });
    }
});
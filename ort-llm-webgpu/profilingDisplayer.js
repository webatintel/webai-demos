"use strict"

/*
Usage:
0. Includes following elements in HTML:
    <link rel="stylesheet" href="sortable.min.css" />
    <link rel="stylesheet" href="profilingDisplayer.css" />
    <script src="sortable.min.js"></script>
    <script src="profilingDisplayer.js"></script>
    <div id="profilingResultsPanel"></div>
1. Set up console.log callback
    let { webgpuProfilingData } = hookConsoleLogForProfilingDisplay(console);
2. Enable and disable ORT WebGPU profiling as needed
    enableWebGPUProfiling(ort);
    // Do some compute in ORT session with WebGPU profiling...
    disableWebGPUProfiling(ort);
    // Do more compute in ORT session without WebGPU profiling...
3. After all needed computation is done, handle and display the profiling results
    // Handle the profiling log and generate the data tables
    const aggregatedTable = generateAggregatedProfilingTable(["Kernel", "Time (ms)", "Percentage (%)"], webgpuProfilingData);
    const dataTable = generateDataTable(["Index", "Kernel", "Time (ms)", "Shape"], webgpuProfilingData);
    // Make profiling results panel visible
    enableProfilingResultsPanel();
    // Display the data tables in the panel
    addDataTable(aggregatedTable, 'Aggregated time');
    addDataTable(dataTable, 'Detailed time');
*/

const displayPrecision = 2;
const unitConversionFactor = 1000000;

function hookConsoleLogForProfilingDisplay(console) {
    let webgpuProfilingData = [];
    let ortProfilingData = [];
    let artifactData = [];
    const hookedProfilingData = { webgpuProfilingData, ortProfilingData, artifactData };

    let originalLog = console.log;
    console.log = (...theArgs) => {
        processConsoleLog(hookedProfilingData, theArgs);
        originalLog.apply(console, theArgs);
    }
    return hookedProfilingData;
}

function enableWebGPUProfiling(ort) {
    ort.env.webgpu.profiling = { mode: 'default' };
}

function disableWebGPUProfiling(ort) {
    ort.env.webgpu.profiling = {};
}

function enableProfilingResultsPanel() {
    document.getElementById('profilingResultsPanel').classList.add('enabled');
}

function clearProfilingResultsPanel() {
    const panel = document.getElementById('profilingResultsPanel');
    const children = [...panel.childNodes];
    children.forEach((node) => panel.removeChild(node));
}

function addDataTable(table, name = 'unnamed') {
    const dataTableWrapper = document.createElement('div');
    dataTableWrapper.className = 'dataTableWrapper show-less';
    dataTableWrapper.dataset['tableName'] = name;

    const wrapperTitle = document.createElement('div');
    wrapperTitle.className = 'dataTableWrapperTitle';
    wrapperTitle.dataset['tableName'] = name;
    wrapperTitle.addEventListener('click', (event) => {
        if (dataTableWrapper.classList.contains('show-less')) {
            dataTableWrapper.classList.replace('show-less', 'show-more');
        } else {
            dataTableWrapper.classList.replace('show-more', 'show-less');
        }
    })

    const wrapperContent = document.createElement('div');
    wrapperContent.className = 'dataTableWrapperContent';
    wrapperContent.appendChild(table);

    dataTableWrapper.appendChild(wrapperTitle);
    dataTableWrapper.appendChild(wrapperContent);

    document.getElementById('profilingResultsPanel').appendChild(dataTableWrapper);
}

function processConsoleLog(hookedProfilingData, args) {
    const { webgpuProfilingData, ortProfilingData, artifactData } = hookedProfilingData;
    let results;
    const content = args[0];
    if (content.startsWith('{"cat"')) {
        results = JSON.parse(content.replace(/,$/, ""));
        let argsResult = results["args"];
        if ("provider" in argsResult) {
            let shape = /(\"input_type_shape.*),\"thread_scheduling_stats\"/.exec(content);
            ortProfilingData.push([
                ortProfilingData.length,
                argsResult["op_name"],
                parseInt(results["dur"]) / unitConversionFactor,
                shape[1],
                argsResult["provider"],
            ]);
        }
    } else if (content.startsWith("[profiling]")) {
        results = /\[profiling\] kernel \"(.*)\" (input.*), execution time\: (\d+) ns/.exec(content);
        let kernelName = "";
        const kernelInfo = results[1].split("|");
        const opType = kernelInfo[1];
        const programName = kernelInfo[3];
        if (opType == programName) {
            kernelName = opType;
        } else {
            kernelName = `${opType}|${programName}`;
        }

        if (results) {
            webgpuProfilingData.push([
                webgpuProfilingData.length,
                kernelName,
                parseInt(results[3]) / unitConversionFactor,
                results[2],
            ]);
        }
    } else if (content.includes("[artifact]")) {
        // console.assert(false, `Unreachable: content.includes("[artifact]")`);
        results = /\[artifact\] key: (.*), programName\: (.*)/.exec(content);
        if (results) {
            artifactData.push([artifactData.length, results[1], results[2]]);
        }
    }
}

function generateDataTable(heads, data) {
    let row, th, td;

    // table
    let table = document.createElement("table");
    table.className = "sortable data-table";
    table.align = "center";
    table.style.width = "80%";
    table.setAttribute("border", "1");

    // thead
    let header = table.createTHead("thead");
    row = header.insertRow(0);
    row.style.fontWeight = "bold";
    for (let head of heads) {
        let th = document.createElement("th");
        th.innerHTML = head;
        row.appendChild(th);
    }

    // tbody
    let tbody = document.createElement("tbody");
    table.appendChild(tbody);
    // rest of line
    for (let i = 0; i < data.length; ++i) {
        let rowInfo = data[i];
        row = tbody.insertRow(i);
        row.onclick = function () {
            // console.log(this);
            this.classList.toggle('highlight');
            // toggleClass(this, "highlight");
        };
        for (let j = 0; j < heads.length; j++) {
            td = row.insertCell(j);
            let cellInfo = rowInfo[j];
            if (heads[j].startsWith("Time")) {
                cellInfo = cellInfo.toFixed(displayPrecision);
            }
            td.innerHTML = cellInfo;
        }
    }

    // tfoot
    let needTfoot = false;
    for (let i = 0; i < heads.length; ++i) {
        if (heads[i].startsWith("Time")) {
            needTfoot = true;
            break;
        }
    }
    if (needTfoot) {
        let tfoot = document.createElement("tfoot");
        table.appendChild(tfoot);
        row = tfoot.insertRow(0);
        row.style.fontWeight = "bold";
        let sums = new Array(heads.length).fill("");
        sums[0] = "Sum";
        for (let i = 0; i < heads.length; ++i) {
            if (!heads[i].startsWith("Time")) {
                continue;
            }

            let sum = 0;
            for (let j = 0; j < data.length; j++) {
                sum += data[j][i];
            }
            sums[i] = sum.toFixed(displayPrecision);
        }
        for (let i = 0; i < heads.length; ++i) {
            td = row.insertCell(i);
            td.innerHTML = sums[i];
        }
    }

    return table;
}

function generateAggregatedProfilingTable(heads, data) {
    let kernelTime = {};
    for (let d of data) {
        let kernel = d[1];
        if (!(kernel in kernelTime)) {
            kernelTime[kernel] = d[2];
        } else {
            kernelTime[kernel] += d[2];
        }
    }
    let totalTime = getSum(Object.values(kernelTime));
    let keys = Object.keys(kernelTime);
    let sortedKernelTime = keys.sort(function (a, b) {
        return kernelTime[b] - kernelTime[a];
    });
    let sortedAggregatedData = [];
    for (let kernel of sortedKernelTime) {
        let time = kernelTime[kernel];
        sortedAggregatedData.push([kernel, time, ((time / totalTime) * 100).toFixed(2)]);
    }

    return generateDataTable(heads, sortedAggregatedData);
}

/*
function renderTask(data) {
    let taskElement = document.createElement("p");
    taskElement.align = "center";
    document.body.appendChild(taskElement);
    taskElement.innerText = `[${task} results]`;

    let resultElement = document.createElement("p");
    resultElement.align = "center";
    document.body.appendChild(resultElement);
    resultElement.id = "result";
    let result = {};

    if (task === "conformance") {
        let _results = [];
        for (let i = 0; i < data[0].length; i++) {
            _results.push([]);
            for (let j = 0; j < data[0][i].length; j++) {
                _results[i].push(compare(data[0][i][j], data[1][i][j], getEpsilons(modelName)));
            }
            _results[i] = `[${_results[i].join(", ")}]`;
        }
        result["result"] = _results.join(", ");

        for (let i = 0; i < data.length; i++) {
            console.info(data[i]);
        }
    } else if (task === "performance") {
        let details = data.join(", ");
        let detailsElement = document.createElement("p");
        document.body.appendChild(detailsElement);
        detailsElement.innerText = details;

        result["first"] = data[0];
        data.shift();
        let totalTime = getSum(data);
        let averageTime = parseFloat((totalTime / data.length).toFixed(2));
        result["average"] = averageTime;
        result["best"] = Math.min(...data);
    }

    if (task === "conformance" || task === "performance") {
        resultElement.innerText = JSON.stringify(result);
        return;
    }

    // profiling
    if (task.includes("Profiling")) {
        resultElement.innerText = `${data[data.length - 1]}ms`;
        if (task === "ortProfiling") {
            generateDataTable(["Index", "Kernel", "Time (ms)", "Shape", "Provider"], ortProfilingData);
        }
        if (task === "webgpuProfiling") {
            generateAggregatedProfilingTable(["Kernel", "Time (ms)", "Percentage (%)"], webgpuProfilingData);
            generateDataTable(["Index", "Kernel", "Time (ms)", "Shape"], webgpuProfilingData);
        }
    }

    if (task === "artifact") {
        generateDataTable(["Index", "Key", "programName"], artifactData);
    }
}
*/
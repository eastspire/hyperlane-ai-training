import os
from flask import Flask, render_template_string, send_from_directory, request, abort

app = Flask(__name__)

# 定义 HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Cloudstudio社区版文件浏览器</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .directory { color: #0066cc; }
        .file { color: #333; }
        .size { width: 100px; }
        .date { width: 180px; }
        .actions { width: 80px; }
        a { text-decoration: none; }
        a:hover { text-decoration: underline; }
        .btn {
            display: inline-block;
            padding: 5px 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
            font-size: 14px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn-download {
            background-color: #008CBA;
        }
        .btn-download:hover {
            background-color: #007B9A;
        }
        .breadcrumb {
            margin-bottom: 20px;
            font-size: 16px;
        }
        .breadcrumb a {
            color: #0066cc;
        }
        .breadcrumb span {
            margin: 0 5px;
        }
    </style>
</head>
<body>
    <h1>文件浏览器</h1>
    
    <div class="breadcrumb">
        <a href="{{ url_for('browse', path='') }}">根目录</a>
        {% for i in range(breadcrumbs|length) %}
            {% if i < breadcrumbs|length - 1 %}
                <span>›</span>
                <a href="{{ url_for('browse', path=breadcrumbs[:i+1]|join('/')) }}">{{ breadcrumbs[i] }}</a>
            {% else %}
                <span>›</span>
                <span>{{ breadcrumbs[i] }}</span>
            {% endif %}
        {% endfor %}
    </div>
    
    <table>
        <tr>
            <th>名称</th>
            <th class="size">大小</th>
            <th class="date">修改日期</th>
            <th class="actions">操作</th>
        </tr>
        {% if path != '' %}
        <tr>
            <td colspan="4"><a href="{{ url_for('browse', path=parent_path) }}">..</a></td>
        </tr>
        {% endif %}
        {% for item in items %}
        <tr>
            <td class="{{ 'directory' if item.is_dir else 'file' }}">
                {% if item.is_dir %}
                    <a href="{{ url_for('browse', path=item.path) }}">{{ item.name }}</a>
                {% else %}
                    {{ item.name }}
                {% endif %}
            </td>
            <td class="size">{{ item.size }}</td>
            <td class="date">{{ item.modified }}</td>
            <td class="actions">
                {% if not item.is_dir %}
                    <a href="{{ url_for('download', path=item.path) }}" class="btn btn-download">下载</a>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


class FileItem:
    def __init__(self, name, path, is_dir, size, modified):
        self.name = name
        self.path = path
        self.is_dir = is_dir
        self.size = size
        self.modified = modified


@app.route("/")
def index():
    return browse("")


@app.route("/browse/<path:path>")
def browse(path):
    # 获取当前目录的绝对路径
    current_dir = os.path.abspath(os.path.join(os.getcwd(), path))

    # 确保用户不能访问当前目录以外的路径
    if not current_dir.startswith(os.getcwd()):
        abort(403)

    # 解析面包屑导航
    breadcrumbs = path.split("/") if path else []

    # 获取父目录路径
    parent_path = "/".join(breadcrumbs[:-1]) if breadcrumbs else ""

    # 获取目录内容
    items = []
    try:
        for entry in os.scandir(current_dir):
            try:
                if entry.is_dir():
                    size = "-"
                else:
                    size = format_size(entry.stat().st_size)
                modified = format_timestamp(entry.stat().st_mtime)
                items.append(
                    FileItem(
                        name=entry.name,
                        path=os.path.join(path, entry.name),
                        is_dir=entry.is_dir(),
                        size=size,
                        modified=modified,
                    )
                )
            except (PermissionError, OSError):
                # 忽略无法访问的文件/目录
                continue
    except (PermissionError, OSError) as e:
        return f"无法访问目录: {str(e)}", 500

    # 按名称排序，目录优先
    items.sort(key=lambda x: (not x.is_dir, x.name.lower()))

    return render_template_string(
        HTML_TEMPLATE,
        items=items,
        path=path,
        parent_path=parent_path,
        breadcrumbs=breadcrumbs,
    )


@app.route("/download/<path:path>")
def download(path):
    # 获取文件的绝对路径
    file_path = os.path.abspath(os.path.join(os.getcwd(), path))

    # 确保用户不能下载当前目录以外的文件
    if not file_path.startswith(os.getcwd()):
        abort(403)

    # 确保路径是文件而不是目录
    if not os.path.isfile(file_path):
        abort(404)

    # 获取文件所在目录和文件名
    directory, filename = os.path.split(file_path)

    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        return f"下载文件失败: {str(e)}", 500


def format_size(size_bytes):
    """将字节大小转换为人类可读的格式"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f} GB"


def format_timestamp(timestamp):
    """将时间戳转换为可读的日期时间格式"""
    from datetime import datetime

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    app.run(debug=True)

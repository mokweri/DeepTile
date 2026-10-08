/**
 * @file graph_loader.cpp
 * @brief Graph handoff JSON loading and schema validation.
 */

#include "deeptile/compiler/graph_loader.h"

#include "deeptile/compiler/model_package.h"
#include "deeptile/compiler/schema_ids.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace deeptile {
namespace graph {
namespace {

void add_error(std::vector<GraphDiagnostic> &diagnostics,
               const std::string &context, const std::string &message) {
  diagnostics.push_back({DiagnosticSeverity::Error, context, message});
}

std::string get_string(const rapidjson::Value &obj, const char *key) {
  if (!obj.HasMember(key) || !obj[key].IsString()) {
    return {};
  }
  return obj[key].GetString();
}

std::vector<int> get_int_array(const rapidjson::Value &obj, const char *key) {
  std::vector<int> out;
  if (!obj.HasMember(key) || !obj[key].IsArray()) {
    return out;
  }
  for (const auto &v : obj[key].GetArray()) {
    if (!v.IsInt()) {
      out.clear();
      return out;
    }
    out.push_back(v.GetInt());
  }
  return out;
}

std::vector<std::string> get_string_array(const rapidjson::Value &obj,
                                          const char *key) {
  std::vector<std::string> out;
  if (!obj.HasMember(key) || !obj[key].IsArray()) {
    return out;
  }
  for (const auto &v : obj[key].GetArray()) {
    if (!v.IsString()) {
      out.clear();
      return out;
    }
    out.push_back(v.GetString());
  }
  return out;
}

std::int64_t product(const std::vector<int> &shape) {
  std::int64_t p = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      return -1;
    }
    p *= dim;
  }
  return p;
}

bool file_size_matches(const std::filesystem::path &path,
                       std::int64_t expected_bytes) {
  if (!std::filesystem::exists(path) ||
      !std::filesystem::is_regular_file(path)) {
    return false;
  }
  return static_cast<std::int64_t>(std::filesystem::file_size(path)) ==
         expected_bytes;
}

bool parse_tensor(const std::string &id, const rapidjson::Value &value,
                  TensorSpec &tensor,
                  std::vector<GraphDiagnostic> &diagnostics) {
  const std::string ctx = "tensor " + id;
  if (!value.IsObject()) {
    add_error(diagnostics, ctx, "tensor record must be an object");
    return false;
  }

  tensor.id = id;
  tensor.kind = get_string(value, "kind");
  tensor.dtype = get_string(value, "dtype");
  tensor.layout = get_string(value, "layout");
  tensor.shape = get_int_array(value, "shape");
  if (value.HasMember("frac") && value["frac"].IsInt()) {
    tensor.frac = value["frac"].GetInt();
  } else {
    add_error(diagnostics, ctx, "missing integer frac");
  }
  if (value.HasMember("file")) {
    if (value["file"].IsString()) {
      tensor.file = std::string(value["file"].GetString());
    } else {
      add_error(diagnostics, ctx, "file must be a string");
    }
  }

  if (tensor.kind.empty()) {
    add_error(diagnostics, ctx, "missing kind");
  }
  if (tensor.dtype != "int8") {
    add_error(diagnostics, ctx, "dtype must be int8");
  }
  if (tensor.layout.empty()) {
    add_error(diagnostics, ctx, "missing layout");
  }
  if (tensor.shape.empty() || product(tensor.shape) <= 0) {
    add_error(diagnostics, ctx, "shape must be a non-empty positive array");
  }
  if ((tensor.kind == "input" || tensor.kind == "param" ||
       tensor.kind == "reference") &&
      !tensor.file.has_value()) {
    add_error(diagnostics, ctx, "file is required for this tensor kind");
  }
  return true;
}

void parse_string_map(const rapidjson::Value &obj,
                      std::map<std::string, std::string> &out) {
  if (!obj.IsObject()) {
    return;
  }
  for (auto it = obj.MemberBegin(); it != obj.MemberEnd(); ++it) {
    if (it->name.IsString() && it->value.IsString()) {
      out[it->name.GetString()] = it->value.GetString();
    }
  }
}

bool parse_node(const rapidjson::Value &value, NodeSpec &node,
                std::vector<GraphDiagnostic> &diagnostics) {
  if (!value.IsObject()) {
    add_error(diagnostics, "node", "node record must be an object");
    return false;
  }
  node.id = get_string(value, "id");
  node.op = get_string(value, "op");
  const std::string ctx = node.id.empty() ? "node" : "node " + node.id;
  if (node.id.empty()) {
    add_error(diagnostics, ctx, "missing id");
  }
  if (node.op.empty()) {
    add_error(diagnostics, ctx, "missing op");
  }
  if (node.op != "conv2d" && node.op != "maxpool2d" &&
      node.op != "gap2d" && node.op != "linear") {
    add_error(diagnostics, ctx, "unsupported op: " + node.op);
  }

  if (value.HasMember("inputs")) {
    parse_string_map(value["inputs"], node.inputs);
  } else {
    add_error(diagnostics, ctx, "missing inputs");
  }
  if (value.HasMember("outputs")) {
    parse_string_map(value["outputs"], node.outputs);
  } else {
    add_error(diagnostics, ctx, "missing outputs");
  }

  if (value.HasMember("attrs") && value["attrs"].IsObject()) {
    const auto &attrs = value["attrs"];
    for (auto it = attrs.MemberBegin(); it != attrs.MemberEnd(); ++it) {
      const std::string key = it->name.GetString();
      if (it->value.IsArray()) {
        std::vector<int> vals;
        for (const auto &v : it->value.GetArray()) {
          if (!v.IsInt()) {
            vals.clear();
            break;
          }
          vals.push_back(v.GetInt());
        }
        node.attrs_vec[key] = vals;
      } else if (it->value.IsInt()) {
        node.attrs_int[key] = it->value.GetInt();
      }
    }
  }

  if (value.HasMember("post_ops") && value["post_ops"].IsObject()) {
    const auto &post = value["post_ops"];
    for (auto it = post.MemberBegin(); it != post.MemberEnd(); ++it) {
      const std::string key = it->name.GetString();
      if (it->value.IsBool()) {
        node.post_ops_bool[key] = it->value.GetBool();
      } else if (it->value.IsInt()) {
        node.post_ops_int[key] = it->value.GetInt();
      } else {
        // A dropped post-op would compile cleanly and then be ignored.
        add_error(diagnostics, "node " + node.id,
                  "post_op " + key + " must be a boolean or an integer");
      }
    }
  }

  return true;
}

void validate_tensor_files(const GraphSpec &graph,
                           std::vector<GraphDiagnostic> &diagnostics) {
  const std::filesystem::path root(graph.root_dir);
  for (const auto &[id, tensor] : graph.tensors) {
    if (!tensor.file.has_value()) {
      continue;
    }
    const std::filesystem::path path = root / *tensor.file;
    const std::int64_t expected = product(tensor.shape);
    if (!file_size_matches(path, expected)) {
      std::ostringstream oss;
      oss << "file " << tensor.file.value() << " missing or size mismatch"
          << " (expected " << expected << " bytes)";
      add_error(diagnostics, "tensor " + id, oss.str());
    }
  }
}

void validate_graph_links(const GraphSpec &graph,
                          std::vector<GraphDiagnostic> &diagnostics) {
  std::set<std::string> available;
  for (const auto &id : graph.inputs) {
    if (graph.tensors.count(id) == 0) {
      add_error(diagnostics, "graph", "input tensor not found: " + id);
    }
    available.insert(id);
  }
  for (const auto &[id, tensor] : graph.tensors) {
    if (tensor.kind == "param") {
      available.insert(id);
    }
  }

  for (const auto &id : graph.outputs) {
    if (graph.tensors.count(id) == 0) {
      add_error(diagnostics, "graph", "output tensor not found: " + id);
    }
  }

  for (const auto &[out, ref] : graph.references) {
    if (graph.tensors.count(out) == 0) {
      add_error(diagnostics, "graph", "reference output not found: " + out);
    }
    auto ref_it = graph.tensors.find(ref);
    if (ref_it == graph.tensors.end()) {
      add_error(diagnostics, "graph", "reference tensor not found: " + ref);
    } else if (ref_it->second.kind != "reference") {
      add_error(diagnostics, "graph", "reference tensor kind is not reference: " + ref);
    }
  }

  for (const auto &node : graph.nodes) {
    const std::string ctx = "node " + node.id;
    for (const auto &[port, tensor_id] : node.inputs) {
      if (graph.tensors.count(tensor_id) == 0) {
        add_error(diagnostics, ctx, "input tensor not found: " + tensor_id);
        continue;
      }
      if (port != "weight" && port != "bias" &&
          available.count(tensor_id) == 0) {
        add_error(diagnostics, ctx,
                  "input tensor is not available before node: " + tensor_id);
      }
    }
    for (const auto &[port, tensor_id] : node.outputs) {
      (void)port;
      if (graph.tensors.count(tensor_id) == 0) {
        add_error(diagnostics, ctx, "output tensor not found: " + tensor_id);
      }
      available.insert(tensor_id);
    }
  }
}

} // namespace

bool load_graph(const std::string &graph_path, GraphSpec &graph,
                std::vector<GraphDiagnostic> &diagnostics) {
  graph = {};
  graph.graph_path = graph_path;
  graph.root_dir = std::filesystem::path(graph_path).parent_path().string();

  if (!validate_model_package(graph_path, diagnostics)) {
    return false;
  }

  std::ifstream input(graph_path);
  if (!input) {
    add_error(diagnostics, "graph", "cannot open graph file: " + graph_path);
    return false;
  }

  rapidjson::IStreamWrapper stream(input);
  rapidjson::Document doc;
  doc.ParseStream(stream);
  if (doc.HasParseError()) {
    std::ostringstream oss;
    oss << "JSON parse error at offset " << doc.GetErrorOffset() << ": "
        << rapidjson::GetParseError_En(doc.GetParseError());
    add_error(diagnostics, "graph", oss.str());
    return false;
  }
  if (!doc.IsObject()) {
    add_error(diagnostics, "graph", "top-level JSON must be an object");
    return false;
  }

  for (const char *key : {"schema", "model", "target", "graph", "tensors",
                         "nodes"}) {
    if (!doc.HasMember(key)) {
      add_error(diagnostics, "graph", std::string("missing top-level field: ") + key);
    }
  }

  graph.schema = get_string(doc, "schema");
  if (!schema::matches(graph.schema, "graph.v1")) {
    add_error(diagnostics, "graph",
              "schema must be deeptile.graph.v1 or tilecnn.graph.v1");
  }

  if (doc.HasMember("model") && doc["model"].IsObject()) {
    parse_string_map(doc["model"], graph.model);
  }

  if (doc.HasMember("target") && doc["target"].IsObject()) {
    const auto &target = doc["target"];
    if (target.HasMember("bitwidth") && target["bitwidth"].IsInt()) {
      graph.target["bitwidth"] = std::to_string(target["bitwidth"].GetInt());
    }
    if (target.HasMember("signed") && target["signed"].IsBool()) {
      graph.target["signed"] = target["signed"].GetBool() ? "true" : "false";
    }
    for (const char *key : {"activation_layout", "conv_weight_layout",
                           "linear_weight_layout", "bias_dtype"}) {
      if (target.HasMember(key) && target[key].IsString()) {
        graph.target[key] = target[key].GetString();
      }
    }
  }

  const auto expect_target = [&](const std::string &key,
                                 const std::string &expected) {
    auto it = graph.target.find(key);
    if (it == graph.target.end() || it->second != expected) {
      add_error(diagnostics, "target",
                key + " must be " + expected);
    }
  };
  expect_target("bitwidth", "8");
  expect_target("signed", "true");
  expect_target("activation_layout", "CHW");
  expect_target("conv_weight_layout", "OIHW");
  expect_target("linear_weight_layout", "OI");
  expect_target("bias_dtype", "int8");

  if (doc.HasMember("graph") && doc["graph"].IsObject()) {
    const auto &g = doc["graph"];
    graph.inputs = get_string_array(g, "inputs");
    graph.outputs = get_string_array(g, "outputs");
    if (g.HasMember("references") && g["references"].IsObject()) {
      parse_string_map(g["references"], graph.references);
    }
  }
  if (graph.inputs.empty()) {
    add_error(diagnostics, "graph", "graph.inputs must be non-empty");
  }
  if (graph.outputs.empty()) {
    add_error(diagnostics, "graph", "graph.outputs must be non-empty");
  }

  if (doc.HasMember("tensors") && doc["tensors"].IsObject()) {
    const auto &tensors = doc["tensors"];
    for (auto it = tensors.MemberBegin(); it != tensors.MemberEnd(); ++it) {
      if (!it->name.IsString()) {
        continue;
      }
      TensorSpec tensor;
      parse_tensor(it->name.GetString(), it->value, tensor, diagnostics);
      graph.tensors[tensor.id] = tensor;
    }
  }

  if (doc.HasMember("nodes") && doc["nodes"].IsArray()) {
    for (const auto &value : doc["nodes"].GetArray()) {
      NodeSpec node;
      parse_node(value, node, diagnostics);
      graph.nodes.push_back(node);
    }
  }
  if (graph.nodes.empty()) {
    add_error(diagnostics, "graph", "nodes must be non-empty");
  }

  validate_tensor_files(graph, diagnostics);
  validate_graph_links(graph, diagnostics);
  return !has_errors(diagnostics);
}

} // namespace graph
} // namespace deeptile
